# 790. Domino and Tromino Tiling

[题目链接](https://leetcode.com/problems/domino-and-tromino-tiling/)

[TOC]

### 思路
![](https://s3-lc-upload.s3.amazonaws.com/users/zhengkaiwei/image_1519539268.png)

* `dp[n] = dp[n-1] + dp[n-2] + 2 * (dp[n-3] + ... + dp[0]) -- E1`
* `dp[n-1] = dp[n-2] + dp[n-3] + 2 * (dp[n-4] + ... + dp[0]) -- E2`
* `E1 - E2:`  `dp[n] - dp[n-1] = dp[n-1] + dp[n-3]`
* `dp[n] = 2*dp[n-1] + dp[n-3]`
#### 动态规划

```cpp
class Solution {
class Solution {
public:
    typedef long long LL;
    int numTilings(int N) {
        int m = 1e9+7;
        vector<LL> dp(N+12,0);
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 5;
        for(int i = 4;i<=N;i++)
        {
            dp[i] = (2 * dp[i-1] %m + dp[i-3]%m)%m;
        }
        
        return dp[N];
    }
};
```

