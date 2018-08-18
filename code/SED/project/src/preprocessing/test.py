import pickle

class A(object):
  
  def __init__(self):

    self.a = 1

class B(object):
  
  def __init__(self):

    self.b = 1


class Test(object):

  def __init__(self):
    self.a = A()
    self.b = B()

  def save(self, path):
    with open(path, 'wb') as f:
      pickle.dump(self, f)

  def load(self, path):
    with open(path, 'rb') as f:
      tmp = pickle.load(f)
      self.a = tmp.a
      self.b = tmp.b

if __name__ == "__main__":
  """
  test = Test()
  test.a.a = 10
  test.b.b = 20
  test.save("/home/gwq/Test.t")
  """
  test = Test()
  print("Before load: a is ", test.a.a, " b is: ", test.b.b)
  test.load("/home/gwq/Test.t")
  print("After load: a is ", test.a.a, " b is: ", test.b.b) 
  