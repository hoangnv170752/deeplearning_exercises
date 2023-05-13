import torch

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
print("**************Exercise 1:")
tensorOne = torch.zeros((20,30,40))
print(tensorOne)
"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
print("**************Exercise 2:")
tensorTwo = torch.rand(4,6)
sumAll = torch.sum(tensorTwo)
sumColumn = torch.sum(tensorTwo, 0)
sumRow = torch.sum(tensorTwo, 1)
print(sumAll)
print(sumColumn)
print(sumRow)
print(tensorTwo)
"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
print("**************Exercise 3:")
tensorThree = torch.tensor([0.1, 0.3, 2.3, 0.45])
tensorThree2 = torch.tensor([0.13, 0.23, 2.33, 0.45])
print("First Tensor: ", tensorThree)
print("Second Tensor: ", tensorThree2)
  
cosi = torch.nn.CosineSimilarity(dim=0)
output = cosi(tensorThree, tensorThree2)
  
print("\n Computed Cosine Similarity: ", output)
"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
print("**************Exercise 4:")
tensorThree = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
tensorThree2 = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
print("First Tensor: \n", tensorThree)
print("Second Tensor: \n", tensorThree2)
cos_1 = torch.nn.CosineSimilarity(dim=0)
output_1 = cos_1(tensorThree, tensorThree2)
print("\n\nComputed Cosine Similarity in dim=0: ",output_1)  
cos_2 = torch.nn.CosineSimilarity(dim=1)
output_2 = cos_2(tensorThree, tensorThree2)
print("\nComputed Cosine Similarity in dim=1: ",output_2)
"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
print("**************Exercise 5:")
tensorFive = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
tuple = (tensorFive[:,0], tensorFive[:,1])
tensorFive1D = torch.cat(tuple)
append = tensorFive1D.unsqueeze(0)
print(tensorFive[:,0])
print(tensorFive[:,1])
print(tensorFive1D)
tensorResize = tensorFive1D.view(3, 4)
print(" Tensor After Resize: \n", tensorResize)
"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
print("**************Exercise 6:")
tensorSix = torch.rand(3, 1080, 1920)
tensorSix2 = torch.rand(3, 720, 1280)
print(tensorSix)
print(tensorSix2)

