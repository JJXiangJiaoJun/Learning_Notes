# 1406. Stone Game III

[题目链接](https://leetcode.com/problems/stone-game-iii/)

[TOC]

### 思路

* 两人依次拿，`Alice`拿到的石头视为`+`，`Bob`拿到的石头视为`-`，`dp[i]`表示以`stones[i]`为起点时，做出最优选择时,`Player1的石头与Player2的石头之差`
* 每次决策都有三种，拿一个，拿两个，拿三个
* 答案为`dp[0]>=0`



#### 动态规划

```cpp
class Solution {
public:
    int dp(vector<int>& value,vector<int>& d,vector<int>& vis,int cur)
    {
        if(cur >= value.size()) return 0;
        if(vis[cur]) return d[cur];
        vis[cur] = 1;
        int sum = 0;
        int &ans = d[cur];
        ans = -(1<<20);
        for(int i = 0;i<3&&(cur+i)<value.size();i++)
        {
            sum += value[cur + i];
            ans = max(ans,sum - dp(value,d,vis,cur + i + 1));
        }
        
        return ans;
    }
    
    string stoneGameIII(vector<int>& stoneValue) {
        int len = stoneValue.size();
        vector<int> d(len + 1,0);
        vector<int> vis(len + 1,0);
        
        int ans = dp(stoneValue,d,vis,0);
        
        if(ans > 0) return "Alice";
        else if(ans < 0) return "Bob";
        else return "Tie";
    }
};
```

