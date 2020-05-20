# 123. Best Time to Buy and Sell Stock III

[题目链接](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/)

[TOC]

### 思路
* `dp[k][i]`,**表示在`prices[i]`这天，完成了`k`次交易所能获得的最大利润**
* 每天决策有，完成交易和不完成交易两种，状态转移方程为
* `dp[k][i] = max(dp[k][i-1],prices[i] + dp[k-1][j] - prices[j]) for 0<j<i`
* 化简一下`dp[k][i] = max(dp[k][i-1],prices[i] + max(dp[k-1][j] - prices[j]))`
* 每次递推即可求得最大值，不用遍历

### 代码



#### 动态规划 

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int len = prices.size();
        if(len <= 1) return 0;
        int K = 2;
        int ans = 0;
        vector<vector<int> >  dp(K+2,vector<int>(len+10,0));
        for(int k = 1; k <= K ;k++)
        {
            int temp_max = dp[k-1][0] - prices[0];
            
            for(int i = 1;i<len;i++)
            {
                dp[k][i] = max(dp[k][i-1],prices[i] + temp_max);
                temp_max = max(temp_max,dp[k-1][i] - prices[i]);
                ans = max(ans,dp[k][i]);
            }
        }
        
        return ans;
        
    }
};
```

