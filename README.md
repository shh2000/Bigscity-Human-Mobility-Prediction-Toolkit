# Bigscity-Human-Mobility-Prediction-Toolkit

datasets类似于loader，不再有extract的部分，里面小json进库，大json后面看看搞个云存储下载的形式，改里面load方法即可。路径在users下面的json写好，users里面的task调用的时候读

docs不用说

test是自己写的接口的调用方式和结果测试，之所以留在库里，主要是方便各自交流的时候清楚调用方式。比如shh_test里面就有datasets的方式（不过路径后面要改成users那个文件的）

evaluate评估

models模型

presentation表示层，把数据筛选接口和预处理部分都放进去

runtimeFiles存放运行时的临时文件、中间结果，相当于cache。！！！！不要在任何其他目录下创建临时文件！！！！

users保存工程配置、写tasks和shell

所有配置以json形式

所有路径相关的都用绝对路径，根目录在users。config。json里面！！！

要加其他目录的话讨论一下，有必要加进去，没必要进ignore。！！！特别要注意\_\_pycache\_\_！！！！,我之前被这个坑过

希望之前最困扰的路径问题能够解决：所有类初始化的时候都要加上根目录绝对路径，users下面第一次读可以用相对路径（毕竟config就在users下面）。users创建其他类时必须填好路径，其他类创建其他类时必须用自己创建时的那个路径去创建其他类

如果还有更好的解决路径问题的方法就改成更好的方法

### 类之前创建关系

shell创建task、evaluate

task创建model、dataset、presentation

dataset、pre、model之前都通过传参获取IO

模型文件保存在runtimefiles里面，这个目录后面改叫cache感觉也可以

过大的预处理结果也放进runtimefiles做缓存（防止炸内存）

## 其他

模型本身最牛逼，所以每个模型单独弄个目录

和模型一对一的presentation也单独目录吧，这个下午讨论一下，我个人感觉放在一起太乱

baseline和通用预处理方法直接放在pre/,models/就ok

注释尽可能中文吧，怕看不懂

异常处理机制尽可能建立完全