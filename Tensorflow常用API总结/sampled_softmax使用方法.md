# 如何实现`sampled_softmax`

```python
if labels is not None:
      responses_vec = responses_vec.view(1, batch_size, -1).expand(batch_size, batch_size, self.vec_dim)
    
    # 这里做attention融合之后输出为`[B,B,H]` 
    # 第一个B表示的是Query，第二个B表示的是`Doc`
    # 也就是Query对每一个Doc Attention输出
    # 每一个Query都对batch中的其他Doc 算了一个向量
    final_context_vec = dot_attention(responses_vec, context_vecs, context_vecs, None, self.dropout)
    final_context_vec = F.normalize(final_context_vec, 2, -1)  # [bs, res_cnt, dim], res_cnt==bs when training

    dot_product = torch.sum(final_context_vec * responses_vec, -1)  # [bs, res_cnt], res_cnt==bs when training
    if labels is not None:
      mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
      
      # 由于第一维是`Query`,那么我们只需要对最后一维进行softmax操作即可
      loss = F.log_softmax(dot_product * 5, dim=-1) * mask
      loss = (-loss.sum(dim=1)).mean()
```
* 目的是对batch中的每一个`Query`,我们都需要计算得到其对batch中所有`Doc`的Score,所以最后我们输出的logits矩阵应该为`[Batch_Size,Batch_Size]`，其中第一个`Batch_size`，表示的是`Query`，也就是每一行数据表示同一个`Query`下所有`Doc`结果，这样我们在`Doc`端计算`softmax`即可。

* 如上使用Tensor`broadcase`功能，每一个`Query`均和batch中的其他`Doc`进行交互，保证以下原则，一般使用`expand_dims`和`tiles`进行扩展
    * 第一维为`Query`表示,比如第一维为`batch_size`，这一维表示的应该是`Query`的`batch_size`，比如每一行数据代表同一个`Query`下对所有`Doc`计算结果
    * 第二维应该为`Doc`表示，这一维表示不同`Query`对这个`Doc`计算结果

* **保证`Query`** 在第一维原因是，根据`sampled_softmax`，我们需要计算同一个`Query`下所有`Doc`概率分布，并进行极大似然估计，那么`Query`在第一维的时候，`Doc`维度就可以看成类别,这样直接在`Doc`维度进行`softmax`运算即可得到概率，从而计算交叉熵损失