from dir_empm.dir_matlab_macros import * ;
import sys ;

print(' %% %% %% %% %% %% %% %% %% ');
print(' %% test_torch_index_0 (2d): ');
print(' %% %% %% %% %% %% %% %% %% ');
n_x = 4; n_y = 3; n_xy = n_x*n_y;
T_xy__ = torch.ones((n_y,n_x));
T_xy_ = torch.flatten(T_xy__);
for ny in range(n_y):
    for nx in range(n_x):
        nxy = nx + ny*n_x;
        vxy = 900 + nx + ny*max(10,n_x);
        T_xy_[nxy] = vxy;
    #end;%for nx in range(n_x)
#end;%for ny in range(n_y);
print(T_xy_);
print(T_xy__);
index_x_ = torch.arange(0,n_x,2);
index_y_ = torch.arange(0,n_y,2);
print(torch.flatten(T_xy__.index_select(0,index_y_).index_select(1,index_x_)));
index_xy_ = matlab_index_2d_0(n_x,index_x_,n_y,index_y_);
print(T_xy_[index_xy_]);
T_xy_[index_xy_] = 999;
print(T_xy_[index_xy_]);
print(T_xy_);
print(T_xy__);

sys.exit(' %% returning after 2d')

print(' %% %% %% %% %% %% %% %% %% ');
print(' %% test_torch_index_0 (3d): ');
print(' %% %% %% %% %% %% %% %% %% ');
n_x = 5; n_y = 4; n_z = 3; n_xyz = n_x*n_y*n_z;
T_xyz___ = torch.ones((n_z,n_y,n_x));
T_xyz_ = torch.flatten(T_xyz___);
for nz in range(n_z):
    for ny in range(n_y):
        for nx in range(n_x):
            nxyz = nx + (ny + nz*n_y)*n_x;
            vxyz = 9000 + nx + (ny + nz*max(10,n_y))*max(10,n_x);
            T_xyz_[nxyz] = vxyz;
        #end;%for nx in range(n_x)
    #end;%for ny in range(n_y);
#end;%for nz in range(n_z);
print(T_xyz_);
print(T_xyz___);
index_x_ = torch.arange(0,n_x,2);
index_y_ = torch.arange(0,n_y,2);
index_z_ = torch.arange(0,n_z,2);
print(torch.flatten(T_xyz___.index_select(0,index_z_).index_select(1,index_y_).index_select(2,index_x_)));
index_xyz_ = matlab_index_3d_0(n_x,index_x_,n_y,index_y_,n_z,index_z_);
print(T_xyz_[index_xyz_]);
T_xyz_[index_xyz_] = 9999;
print(T_xyz_[index_xyz_]);
print(T_xyz_);
print(T_xyz___);

print(' %% %% %% %% %% %% %% %% %% ');
print(' %% test_torch_index_0 (4d): ');
print(' %% %% %% %% %% %% %% %% %% ');
n_x = 6; n_y = 5; n_z = 4; n_w = 3; n_xyzw = n_x*n_y*n_z*n_w;
T_xyzw____ = torch.ones((n_w,n_z,n_y,n_x));
T_xyzw_ = torch.flatten(T_xyzw____);
for nw in range(n_w):
    for nz in range(n_z):
        for ny in range(n_y):
            for nx in range(n_x):
                nxyzw = nx + (ny + (nz + nw*n_z)*n_y)*n_x;
                vxyzw = 90000 + nx + (ny + (nz + nw*max(10,n_z))*max(10,n_y))*max(10,n_x);
                T_xyzw_[nxyzw] = vxyzw;
            #end;%for nx in range(n_x)
        #end;%for ny in range(n_y);
    #end;%for nz in range(n_z);
#end;%for nw in range(n_w);
print(T_xyzw_);
print(T_xyzw____);
index_x_ = torch.arange(0,n_x,2);
index_y_ = torch.arange(0,n_y,2);
index_z_ = torch.arange(0,n_z,2);
index_w_ = torch.arange(0,n_w,2);
print(torch.flatten(T_xyzw____.index_select(0,index_w_).index_select(1,index_z_).index_select(2,index_y_).index_select(3,index_x_)));
index_xyzw_ = matlab_index_4d_0(n_x,index_x_,n_y,index_y_,n_z,index_z_,n_w,index_w_);
print(T_xyzw_[index_xyzw_]);
T_xyzw_[index_xyzw_] = 99999;
print(T_xyzw_[index_xyzw_]);
print(T_xyzw_);
print(T_xyzw____);
