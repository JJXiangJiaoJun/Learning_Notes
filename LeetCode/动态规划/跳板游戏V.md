# 1340. Jump Game V

[题目链接](https://leetcode.com/problems/jump-game-v/)

[TOC]

### 思路
* 使用记忆化搜索`dp(i)`表示从`i`开始**能够跳的最多个数**
* 相当于在DAG上找到一个最长路径
* 可以使用单调栈预处理出，每个点能向最左，最右跳到的点数。

### 代码

#### 动态规划

```cpp
class Solution {
public:
    int memo[1000+10];
    int left[1000+10];
    int right[1000+10];
    int vis[1000+10];
    
    int dp(int idx,int &d,int len)
    {
        if(vis[idx]) return memo[idx];
        vis[idx] = 1;
        int &ans = memo[idx];
        
        ans = 0;
        for(int k=idx-d;k<=idx+d;k++)
        {
            if(k<0||k>=len||k==idx||k<left[idx]||k>right[idx]) continue;
            ans = max(ans,dp(k,d,len));
        }
        ans = ans +1;
        return ans;
    }
    
    int maxJumps(vector<int>& arr, int d) {
        stack<int> st;
        memset(vis,0,sizeof(vis));
        int len = arr.size();
        for(int i=0;i<len;i++)
        {
            while(!st.empty()&&arr[st.top()]<arr[i]) st.pop();
            left[i] = st.empty()?0:st.top()+1;
            st.push(i);
        }
        
        while(!st.empty()) st.pop();
        for(int i=len-1;i>=0;i--)
        {
             while(!st.empty()&&arr[st.top()]<arr[i]) st.pop();
             right[i] = st.empty()?len-1:st.top()-1;
             st.push(i);   
        }
        
        
        int ans = 0;
        for(int i=0;i<len;i++)
            if(!vis[i]) ans = max(ans,dp(i,d,len)); 
        return ans;
        
    }
};
```

