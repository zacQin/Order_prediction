import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Library/Fonts/Songti.ttc')

font_size =8
fig_size = (18,16)

names = (u'9月9日',u'9月10日教师节')
name_list = (u'北 京',u'上 海',u'成 都',u'杭 州',u'深 圳',u'广 州',u'全 国')

order_9_9=[63788,22269,8948,4506,4018,3792,174739]
order_9_10=[89245,33613,14309,7993,5504,5256,223813]
# x=list(range(len(order_9_9)))
total_width ,n = .8,2
width = total_width/n
x=np.arange(len(order_9_10))
fig,ax=plt.subplots()

b1=ax.bar(x,order_9_9, label='2018_9_9',width =width,fc='cyan')
b2=ax.bar(x+width,order_9_10,width =width,label='2018_9_10',tick_label = name_list ,fc='b')
#label='9月10日教师'
for rect in b1+b2:
    h=rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2,h,'%d'%int(h),ha='center',va='bottom')
plt.title(u'2018年9月10日教师节订单预测', fontproperties=font)
plt.xticks(x + width, name_list, fontproperties=font)
plt.legend()
plt.show()