# 1388. Pizza With 3n Slices

[题目链接](https://leetcode.com/problems/pizza-with-3n-slices/)

[TOC]

### 思路
* 首先我们不能同时选择`0或者n-1`，所以我们分成两种情况，`[0...n-2]`以及`[1...n-1]`
* 定义`dp[i][j]`为，从`[0..i-2]`，或者`[1...i-1]`，挑选`j`块时，能够获得的最大值
    * ` dp[i][j] = max(dp[i - 1][j], dp[i - 2][j - 1] + slices[i - 1 - !l]);` 
* **这种两头都有影响的可以考虑递推，这样就只有一头有影响了**




### 代码

#### 动态规划

```cpp
class Solution {
private:
    int maxSizeSlices(const vector<int> &slices, int l) {
        int n = slices.size();
        vector<vector<int>> dp(n + 1, vector<int>(n / 3 + 1, 0));
        
        for(int i = 2; i < n + 1; i++)
            for(int j = 1; j <= n / 3; j++)
                dp[i][j] = max(dp[i - 1][j], dp[i - 2][j - 1] + slices[i - 1 - !l]);
       
        return dp[n][n / 3];
    }
    
public:
    int maxSizeSlices(const vector<int>& slices) {
        return max(maxSizeSlices(slices, 0), maxSizeSlices(slices, 1));
    }
};
```

