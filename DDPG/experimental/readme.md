（（中止））
Matthias Plappert 氏の議論，コード
https://groups.google.com/forum/#!msg/keras-users/Rowv98EMk28/YE0HJCf7BwAJ
https://gist.github.com/matthiasplappert/60c62a194cdc37a97e9d
https://github.com/matthiasplappert/keras-rl/blob/master/rl/agents/ddpg.py

を参考に，actorの勾配dQ/dθ (θはactorのパラメータ．)の計算を
dQ/da * da/dθ とせずに直接計算する試み．ついでにすべてKeras化．

optimizer.get_updates(loss, params) 中で
optimizer.get_gradients(loss, params) が実行される
https://github.com/keras-team/keras/blob/b187ac51e0c9ae6fab4aabfb12ceed57c83b12ba/keras/optimizers.py#L162
が，その返り値がNoneになる模様．(エラーメッセージ中の変数 g)
そもそも共有ネットワークじゃないから勾配あるわけないか．
