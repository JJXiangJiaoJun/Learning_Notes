# 629. K Inverse Pairs Array

[题目链接](https://leetcode.com/problems/k-inverse-pairs-array/)

[TOC]

### 思路

* 定义`dp[i][j]`，为考虑`[1,i]`数字，逆序对为`j`时候的排列个数
* 比如我们已经排列好`1...4` 
    * `5 x x x x`有4个新的逆序对
    * `x 5 x x x`有3个新的逆序对
    * .....
* 递推式为`dp[i][j] = dp[i-1][j] + dp[i-1][j-1] + dp[i-1][j-2] + ..... +dp[i-1][j - i + 1]`
* 但是这样每次加都需要遍历求和
* 我们注意到`dp[i][j-1] = dp[i-1][j-1] + dp[i-1][j-1] + dp[i-1][j-2] + ..... +dp[i-1][j - i ]`
* 可以用`dp[i][j-1]`来计算`dp[i][j]`，减少时间复杂度
* **以后这种需要求和的都可以做优化**





#### 动态规划

```cpp
class Solution {
public:
    typedef long long LL;
    
    int kInversePairs(int n, int k) {
        
        vector<vector<LL>> dp(n + 3 , vector<LL>(k + 3 ,0));
        //vector<vector<LL>> sum(n + 3 , vector<LL>(k + 3 ,0));     
        LL mod = 1e9+7;
        dp[0][0] = 1;
        for(int i = 1 ;i <= n ;i++)
        { 
            dp[i][0] = 1;
            for(int j = 1; j <= k ;j++)
            {
                dp[i][j] = (dp[i][j-1] + dp[i-1][j])%mod;
                
                if(j >= i)
                {
                    dp[i][j] = (dp[i][j] - dp[i-1][j-i] + mod) % mod;
                }
            }
        }
        return dp[n][k];
    }
};
```

