  
import math
class Channel_Variations(object):
  def fibonacci(self, depth):
    f = []
    for i in range(1,depth+1):
        num = ((1+math.sqrt(5))**i) - ((1-math.sqrt(5))**i)
        den = 2**i * (math.sqrt(5))
        f.append(int(num/den))
    return(f)

  def logistic(self, a,x):
    return a * x * (1-x)
  
  def get(self, in_channels = 3, n_blocks=5,  depth = 5, ratio = 0.618, is_reverse = False):
    blocks_list = self.fibonacci(n_blocks)
    channel_list =[in_channels]
    ratio_list = [ratio]
    for block in blocks_list:
        depth_ = depth
        ratio_ = ratio 
        while depth_ > 0:
            val = int( (block * ratio_ * (1 - ratio_))*100)
            channel_list.append(val)
            ratio_ = self.logistic(2.4, ratio_)
            depth_ -= 1
            ratio_list.append(ratio_)
    return channel_list   
