from dir_empm.dir_matlab_macros import * ;

print(' %% test_torch_index_0:');
n_x = 4; n_y = 5; n_xy = n_x*n_y;
T_xy__ = torch.ones((n_y,n_x));
T_xy_ = torch.flatten(T_xy__);
for ny in range(n_y):
    for nx in range(n_x):
        nxy = nx + ny*n_x;
        T_xy_[nxy] = nxy;
    #end;%for nx in range(n_x)
#end;%for ny in range(n_y);
print(T_xy_);
print(T_xy__);
index_x_ = torch.arange(0,n_x,2);
index_y_ = torch.arange(0,n_y,2);
print(torch.flatten(T_xy__.index_select(0,index_y_).index_select(1,index_x_)));

index_y__,index_x__ = torch.meshgrid(index_y_,index_x_,indexing='ij');
index_xy__ = index_x__ + index_y__*n_x;
index_xy_ = torch.flatten(index_xy__);
print(T_xy_[index_xy_]);
T_xy_[index_xy_] = 999;
print(T_xy_[index_xy_]);
print(T_xy_);
print(T_xy__);
