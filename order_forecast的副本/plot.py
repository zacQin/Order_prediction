
import matplotlib.pyplot as plt
import numpy as np

font_size =10
fig_size = (8,6)

names = (u'9月9日',u'9月10日教师节')
name_list = (u'北京',u'上海',u'成都',u'杭州',u'深圳',u'广州',u'全国')

order_9_9=[63788,22269,8948,7788,4018,7788,174739]
order_9_10=[89245,33613,14309,7788,5504,7788,223813]
# x=list(range(len(order_9_9)))
total_width ,n = .8,2
width = total_width/n
x=np.arange(len(order_9_10))
fig,ax=plt.subplots()

b1=ax.bar(x,order_9_9, width =width,label = '9月9日',fc='y')
b2=ax.bar(x+width,order_9_10,width =width,label='9月10日教师',tick_label = name_list ,fc='r')

for rect in b1+b2:
    h=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2,h,'%d'%int(h),ha='center',va='bottom')

plt.legend()
plt.show()