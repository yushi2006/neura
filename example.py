import neura

a = neura.Tensor([1.0, 2.0, 4.0])
b = a.log().sum()
print(b)

b.backward()

print(a.grad)
