import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import activation_getter


class perec(nn.Module):
	"""
	Convolutional Sequence Embedding Recommendation Model (Caser)[1].

	[1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

	Parameters
	----------

	num_users: int,
					Number of users.
	num_items: int,
					Number of items.
	model_args: args,
					Model-related arguments, like latent dimensions.
	"""

	def __init__(self, num_users, num_items, model_args):
		super(perec, self).__init__()
		self.args = model_args

		# init args
		L = self.args.L
		dims = self.args.d
		self.n_h = self.args.nh
		self.n_v = self.args.nv
		self.drop_ratio = self.args.drop
		self.ac_conv = activation_getter[self.args.ac_conv]
		self.ac_fc = activation_getter[self.args.ac_fc]

		# user and item embeddings
		self.user_embeddings = nn.Embedding(num_users, dims)
		self.item_embeddings = nn.Embedding(num_items, dims)

		# personalized gate
		self.feature_gate_item = nn.Linear(dims, dims)
		self.feature_gate_user = nn.Linear(dims, dims)

		# merge gate
		self.merge_gate1 = nn.Linear(dims, dims)
		self.merge_gate2 = nn.Linear(dims, dims)

		# self attetnion
		self.has_residual = True
		self.block_shape = [dims, dims]
		self.w_qs = nn.Linear(dims, dims)
		self.w_ks = nn.Linear(dims, dims)
		self.w_vs = nn.Linear(dims, dims)

		self.sequence_to_single = Variable(torch.zeros(
			dims, 1).type(torch.FloatTensor), requires_grad=True)
		self.sequence_to_single = torch.nn.init.xavier_uniform_(
			self.sequence_to_single)
		self.V_res_linear = nn.Linear(dims, dims)
		self.softmax = nn.Softmax(dim=2)

		self.layer_norm = nn.LayerNorm(dims)

		# fully-connected layer
		# self.fc1_dim_v = self.n_v * dims
		# self.fc1_dim_h = self.n_h * len(lengths)
		# fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
		# W1, b1 can be encoded with nn.Linear
		# self.fc1 = nn.Linear(fc1_dim_in, dims)
		# W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
		self.W2 = nn.Embedding(num_items, dims+dims)
		self.b2 = nn.Embedding(num_items, 1)

		# dropout
		self.dropout = nn.Dropout(self.drop_ratio)

		# weight initialization
		self.user_embeddings.weight.data.normal_(
			0, 1.0 / self.user_embeddings.embedding_dim)
		self.item_embeddings.weight.data.normal_(
			0, 1.0 / self.item_embeddings.embedding_dim)
		self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
		self.b2.weight.data.zero_()

		self.cache_x = None

	def forward(self, seq_var, user_var, item_var, for_pred=False):
		"""
		The forward propagation used to get recommendation scores, given
		triplet (user, sequence, targets).

		Parameters
		----------

		seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
						a batch of sequence
		user_var: torch.LongTensor with size [batch_size]
						a batch of user
		item_var: torch.LongTensor with size [batch_size]
						a batch of items
		for_pred: boolean, optional
						Train or Prediction. Set to True when evaluation.
		"""

		# Embedding Look-up
		item_embs = self.item_embeddings(seq_var)  # use unsqueeze() to get 4-D
		user_emb = self.user_embeddings(user_var).squeeze(1)

		# feature gating
		gate = torch.sigmoid(self.feature_gate_item(
			item_embs) + self.feature_gate_user(user_emb).unsqueeze(1))
		gated_item = item_embs*gate
		z = torch.mean(gated_item, dim=1)

		# preference excitation item-item relation
		queries = item_embs*(1+torch.sigmoid(z).unsqueeze(1))
		keys = item_embs*(1+torch.sigmoid(z).unsqueeze(1))
		self_attention_item = item_embs
		for i in range(len(self.block_shape)):
			self_attention_item = self.multihead_attention(queries=queries,
                                                  keys=keys,
														   values=self_attention_item,
														   num_units=self.block_shape[i],
														   num_heads=1,
														   has_residual=self.has_residual)
		# print("self_attention_item", self_attention_item.size())
		
		merge_gate_score = torch.sigmoid(self.merge_gate1(
			self_attention_item)+self.merge_gate2(gated_item))
		
		return_score= torch.mean(merge_gate_score,dim=2)
		# print("return_score", return_score.size())
		return_score = torch.mean(return_score, dim=1)
		# print("return_score", return_score.size())
		# print("return_score", return_score)
		merge_item = merge_gate_score*self_attention_item + \
			(1-merge_gate_score)*gated_item
		# print("merge_item",merge_item.size())

		# print("z", z.size())   # [512,100]
		# print("self_attention_item", self_attention_item.size())   # [512,5,100]
		q = torch.mean(merge_item, dim=1)
		x = torch.cat([q, user_emb], 1)
		x = self.dropout(x)

		w2 = self.W2(item_var)
		b2 = self.b2(item_var)

		if for_pred:
			w2 = w2.squeeze()
			b2 = b2.squeeze()
			res = x.mm(w2.t()) + b2
		else:

			res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()


		return res

	def multihead_attention(self, queries,
							keys,
							values,
							num_units=None,
							num_heads=1,
							has_residual=True):
		if num_units is None:
			num_units = queries.size()[-1]

		# linear  projections
		# Q = self.ac_fc(self.w_qs(queries))
		# K = self.ac_fc(self.w_ks(keys))
		# V = self.ac_fc(self.w_vs(values))

		Q = self.w_qs(queries)
		K = self.w_ks(keys)
		V = self.w_vs(values)
		# print("Q", Q.size())
		# print("K", K.size())
		# print("V", V.size())

		if has_residual:
			V_res = self.ac_fc(self.V_res_linear(values))

		# Split and concat
		Q_ = torch.cat(Q.chunk(num_heads, dim=2), dim=0)
		K_ = torch.cat(K.chunk(num_heads, dim=2), dim=0)
		V_ = torch.cat(V.chunk(num_heads, dim=2), dim=0)

		# print("Q_", Q_.size())
		# print("K_", K_.size())
		# print("V_",V_.size())
		# Multiplication
		outputs = torch.bmm(Q_, K_.permute(0, 2, 1))

		# scale
		outputs = outputs / (K_.size()[-1]**0.5)

		# Activation
		weights = self.softmax(outputs)

		# print(weights)


		weights = self.dropout(weights)

		# Weighted sum
		outputs = torch.bmm(weights, V_)

		# Restore shape
		outputs = torch.cat(outputs.chunk(num_heads, dim=0), dim=2)

		# residual connection
		outputs = self.ac_fc(outputs)
		# outputs=self.dropout(outputs)
		if has_residual:
			outputs += V_res
		# outputs = self.ac_fc(outputs)
		output = self.layer_norm(outputs)

		# print("output",output.size())

		return output


