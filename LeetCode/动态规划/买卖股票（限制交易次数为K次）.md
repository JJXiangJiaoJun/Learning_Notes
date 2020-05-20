# 188. Best Time to Buy and Sell Stock IV

[题目链接](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/)

[TOC]

### 思路
* `dp[k][i]`,**表示在`prices[i]`这天，已经完成了`k`次交易所能获得的最大利润**
* 每天决策有，完成交易和不完成交易两种，状态转移方程为
* `dp[k][i] = max(dp[k][i-1],prices[i] + dp[k-1][j] - prices[j]) for 0<j<i`
* 化简一下`dp[k][i] = max(dp[k][i-1],prices[i] + max(dp[k-1][j] - prices[j]))`
* 每次递推即可求得最大值，不用遍历
* 注意判断`k > len / 2`时，每天都可以选择进行交易，只要利润大于0，那么就进行交易

### 代码



#### 动态规划 

```cpp
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int len = prices.size();
       
        if(len <= 1) return 0;
        
        int ans  = 0;
        
        if (k>len/2){ // simple case
            //int ans = 0;
            for (int i=1; i<len; ++i){
                ans += max(prices[i] - prices[i-1],0);
            }
            return ans;
        }
        
         vector<vector<int>> dp(k + 2, vector<int>(len + 2 , 0));
        for(int i = 1; i <= k ; i++)
        {
            int temp_max = dp[i-1][0] - prices[0];
            for(int j = 1;j < len ; j++)
            {
                dp[i][j] = max(dp[i][j-1] , prices[j] + temp_max);
                temp_max = max(temp_max,dp[i-1][j] - prices[j]);
                ans = max(ans,dp[i][j]);
            }
        }
        
        return ans;
        
        
    }
};


```

