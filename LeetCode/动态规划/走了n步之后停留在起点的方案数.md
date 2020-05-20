# 1269. Number of Ways to Stay in the Same Place After Some Steps

[题目链接](https://leetcode.com/problems/number-of-ways-to-stay-in-the-same-place-after-some-steps/)

[TOC]

### 思路
* `dp[i][j]`表示走了`i`步后停留在下标`j`的方案数
    * 每次有三个决策，向左、停留、向右
    * `dp[i][j] = dp[i-1][j-1] + dp[i-1][j] + dp[i-1][j+1]`
* **注意!!!**，可以进行优化，因为走到的最大下标就是一直向左走`arrLen = min(arrLen,steps)`


### 代码

#### 动态规划

```cpp
class Solution {
public:
    typedef long long LL;
    LL dp[2][1000000];
    int numWays(int steps, int arrLen) {
        int cur_idx = 0;
        int m = 1e9+7;
        arrLen = min(arrLen,steps);
        //vector<vector<LL>> dp(2,vector<LL>(arrLen,0));
         //memset(dp,0,sizeof(dp));
        dp[cur_idx][0] = 1;
        for(int i=1;i<=steps;i++)
        {   
            int new_idx = cur_idx==1?0:1;
           
            for(int j=0;j<arrLen;j++)
            {
                dp[new_idx][j] = 0;
                if(j!=0) dp[new_idx][j]  = (dp[cur_idx][j-1] +dp[new_idx][j])%m;
                if(j!=arrLen-1)   dp[cur_idx^1][j]  = (dp[cur_idx][j+1] +dp[new_idx][j])%m;
                 dp[new_idx][j]  = (dp[cur_idx][j] +dp[new_idx][j])%m;
            }
            cur_idx = new_idx;
        }
        
        return dp[cur_idx][0];
    }
};
```

