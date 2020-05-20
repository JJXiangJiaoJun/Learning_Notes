[TOC]
# 51. N-Queens
[题目链接](https://leetcode.com/problems/n-queens/)

### 思路
* 皇后不能同时在**一行、一列、斜对角中**
* 斜对角判断用**`cur_c - cur_r == pre_c - pre_r || cur_c + cur_r == pre_c + pre_r`**
* 分别是**判断主对角线以及副对角线**
* 用`vis[2][]`分别记录每列，主、副对角线是否占用
* 回溯法

### 代码

```cpp
class Solution {
public:
    vector<vector<string>> ans;
    
    void dfs(int cur,int &n,vector<vector<int>>  &vis,vector<string> &temp)
    {
        if(cur>=n) {
            ans.push_back(temp);
            return;
        }
        
        for(int j=0;j<n;j++)
        {
            if(vis[0][j]==0&&vis[1][j+cur]==0&&vis[2][j-cur+n]==0)
            {
                vis[0][j] = vis[1][j+cur] = vis[2][j-cur+n] = 1;
                temp[cur][j] = 'Q';
                dfs(cur+1,n,vis,temp);
                temp[cur][j] = '.';
                vis[0][j] = vis[1][j+cur] = vis[2][j-cur+n] = 0;
            }
            
        }
    }
    
    vector<vector<string>> solveNQueens(int n) {
            vector<vector<int>> vis(3,vector(n+n+2,0));
            vector<string> temp(n,string(n,'.'));
            
            dfs(0,n,vis,temp);
            return ans;
    }
};
```

