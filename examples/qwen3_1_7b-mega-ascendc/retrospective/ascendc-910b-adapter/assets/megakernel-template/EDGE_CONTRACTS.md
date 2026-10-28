# 每个stage的交接壳：修改计算时保留什么

符号：`R=8*layer`，`epoch=tag`，AIC seq row=`24+cid`，AIV seq row=`cid`；固定地址用不同round分离，跨token tag递增512。seq与data是不同对象，收到seq不自动证明任意data可见。

| 逻辑边 | writer→reader | data/ready | producer必须完成 | consumer与reuse要求 |
|---|---|---|---|---|
| RMS1→QKV | AIV sub0→同组AIC | xn1P / AIV R+0 | pack、MTE3及source保护、seq | AIC先wait再load A；独立B可提前ring_fill |
| QKV→QKnorm/RoPE | 分片AIC→对应Q/K/V AIV | qkv_out / AIC R+0 | Fixpipe、FIX_S、sentinel、seq | 根据Q/K/V列覆盖轮询所需AIC，不假设同cid覆盖全部 |
| KVappend/Q2→QK | AIV sub0→AIC | KcP/VcP/q2P / AIV R+2 | 唯一sidOwner写新KV行；DMA与UB生命周期完成 | QK读有效前缀；padding保持零；每slice都要publish |
| QK→softmax | AIC→对应AIV | scores16 / PR_SCORES | 每4096位置group末尾数据marker落地 | AIV只读已发布group；ping-pong前等vector reader |
| softmax→PV | AIV sub0→AIC | probs2P / PR_PROBS | compact probs写GM并确认后seq | PV先读独立V，再等对应probs grain；grain tail有效 |
| partial→combine | AIC/AIV→slice leader | partAcc/part_l / AIC R+4、AIV R+3 | Acc落地；denominator发布readback | 先认证live Acc，后读denominator；dead slice不得参与 |
| combine→O | 各head leader→所有O核 | attnP / AIV R+5 | head数据pack落地，leader发布 | AIC等8个head leader；不是等所有24个seq |
| O→residual/RMS2 | 活跃AIC→AIV | o_out / AIC R+5 | 每活跃slab落地后seq | 只等0..21；读H有效元素；warm不能覆盖UB |
| RMS2→GU | AIV sub0→同组AIC | xn2P / AIV R+6 | pack/MTE3/seq | GU ring可先预取权重；输入必须ready |
| GU→SiLU | 每个AIC→同组AIV | gu_out / AIC R+6 | **live row每128列tile**落地后发布 | paired gate/up同producer；legacy mode producer不同 |
| SiLU→Down | AIV sub0→全部Down AIC | hP / AIV R+7 | 对应256元素pack/MTE3/seq | 按0..7、8..15、16..23三组wait/load |
| Down→residual | 活跃AIC→AIV | d_out / AIC R+7 | active slab落地后发布 | resid后下一层；空AIC22/23也等后两组SiLU reader退休才能重臂GU |
| final norm→LM | AIV sub0→同组AIC | xnfP / AIV224 | final RMS pack与seq | LM wait自己的输入；ring独立权重可预取 |
| LM→argmax | AIC→同组AIV | logits / PR_LM或AIC224 | 实际live-logit grain marker/FIX_S/seq | 消费自己span；mask补齐词表；UB reader完成后reload |
| argmax→host | AIV sub0→host | part_max/part_idx | MTE3与完整kernel完成 | 不把未经认证的Fixpipe host读回当真值 |

## buffer的所有权不是“谁分配”

- ring L1 slot：MTE2 writer、MTE1 reader；ready是MTE2_MTE1，free是MTE1_MTE2。
- A-group L1：8个K chunk；读完当前组的LoadData后，才能加载下一组。
- L0A/L0B：MTE1写、Mmad读；L0C：Mmad写、Fixpipe读。FIX_M只代表源可复用；目的可见性另用FIX_S/marker。
- UB：MTE2/vector/scalar/MTE3都可能访问。每个方向分别配event，不能靠一个PipeBarrier替代。
- live GU marker：arm本身是一次写，须等待上一层reader完成。不能只证明上次writer完成。
- maxima history：两个bank交替，防止前层读取与本层覆写相撞。

## 当前event预算

ring双向1,2,3,4,6,7，A-group5；attention ping-pong双向0/1。attention用1前ring必须drain。M_MTE1、M_FIX、FIX_M是各自event类型的独立方向；不要把同一个数字在不同类型的含义混起来，也不要把同一类型同ID分给两个outstanding任务。

V_S/S_V保护scalar GetValue和vector scratch；MTE2_S保护GM加载后的scalar读；V_MTE2保护reader结束再reload；MTE3_V或MTE3_MTE2保护UB源复用。`FetchEventID`在该版本只是peek，不能用连续两次调用来分配双缓冲ID。

新增stage时在本表加一行并画出ready与free两条边；删stage时删除无人消费的publish前，先确认下游没有依赖它提供的可见性或reader-ack。
