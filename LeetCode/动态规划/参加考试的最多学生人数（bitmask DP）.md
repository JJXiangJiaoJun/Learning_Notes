# 1349. Maximum Students Taking Exam

[题目链接](https://leetcode.com/problems/maximum-students-taking-exam/)

[TOC]

### 思路

* `dp[i][j]`,表示当前为第`i`行，坐的学生集合为`j`时，最多能做的数量。`j`某一位为1表示当前位置坐了学生
* **bitmask中常用的操作**
    *  **(x>>i)&1**和**x&(1<<i)**，取出第`i`个状态
    *  **(x&y) == x**，判断x是否为y的**子集**
    *  **(x&(x>>1)) == 0**，判断是否有相邻的状态
*  这题中，我们的状态转移方程如下：
    * `dp[i][mask] = max(dp[i - 1][mask']) + number of valid bits(mask)`  
* **`mask'`**，是`i-1`排的学生安排，为了防止作弊，则需要
    * `(mask & (mask' >> 1)) == 0` ,没有坐在左上角的学生
    * `((mask >> 1) & mask') == 0,`，没有坐在右上角的学生
* `__builtin_popcount(mask)`，可以快速计算`mask`中有多少个1

#### 动态规划

```cpp
class Solution {
public:
    int maxStudents(vector<vector<char>>& seats) {
        int m = seats.size();
        int n = seats.back().size();
        
        vector<int> validation(m,0);
        for(int i = 0 ;i < m; i++)
        {
            int cur = 0;
            for(int j = 0 ; j < n ; j ++)
            {
                cur = (cur<<1)|(seats[i][j]=='.');
            }
            
            validation[i] = cur;
        }
        
        vector<vector<int> >  dp(m+1,vector<int>((1<<n),-1));
        
        dp[0][0] = 0;
        
        for(int i = 1;i <= m ; i++)
        {
            int valid = validation[i-1];
            
            for(int j = 0; j < (1<<n);j++)
            {
                if((j&valid)==j&&!(j&(j>>1)))
                {
                    
                    for(int k = 0; k < (1<<n) ; k++)
                    {
                        if(!((j>>1)&k)&&!(j&(k>>1))&&dp[i-1][k]!=-1)
                        {
                            dp[i][j] = max(dp[i][j],dp[i-1][k]+__builtin_popcount(j));
                        }
                    }
                    
                }
            }
        }
        
        int ans = 0;
        for(int i = 0; i< (1<<n);i++)
            ans = max(ans,dp[m][i]);
        return ans;
    }
};
```

