import h5py

f = h5py.File("0001.h5",'r')

for key in f.keys():
    # print(f[key].name)
    # print(f[key].shape)

    print(f[key][:])
    # print(f[key].value)