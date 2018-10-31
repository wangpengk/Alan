# 导入tensorflow模块
import tensorflow as tf
# 定义变量
state = tf.Variable(0)
# 定义加法
new_value = tf.add(state, 1)
# 更新变量
update = tf.assign(state, new_value)
# 初始化变量
init = tf.global_variables_initializer()
# 定义会话
result = 0
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        result += sess.run(update)
    print(result)


