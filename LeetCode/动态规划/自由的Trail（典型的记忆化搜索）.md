# 514. Freedom Trail

[题目链接](https://leetcode.com/problems/freedom-trail/)

[TOC]

### 思路
* 以后考虑`dp`都考虑两种情况
    * dfs+memo，也就是记忆化搜索
    * 递推
* 记忆化搜索的key要确定好




### 代码

#### 动态规划

```cpp
class Solution {
public:
   
    
    int dp(string &ring,string &key,int cur,int idx,unordered_map<int,int> &rk2cnt)
    {
        if(cur >= key.length()) return 0;
        int search_k = cur*100 + idx;
        if(rk2cnt.count(search_k)) return rk2cnt[search_k];
      
        int ans = INT_MAX;
        
        for(int i=0;i<ring.length();i++)
        {
            if(ring[i]!=key[cur]) continue;
            int diff = abs(i - idx);
            int distance = min(diff,((int)ring.length()-diff)) + dp(ring,key,cur+1,i,rk2cnt);
           // ans = min(ans,min(diff,((int)ring.length()-diff)) + dp(ring,key,cur+1,i));
            ans = min(ans,distance);
        }
        rk2cnt[search_k]= ans;
        return ans;
    }
    
    int findRotateSteps(string ring, string key) {
        unordered_map<int,int> rk2cnt;
        return dp(ring,key,0,0,rk2cnt) + key.length();
    }

};
```

