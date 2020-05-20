# 40. Combination Sum II
[TOC]

[题目链接](https://leetcode.com/problems/combination-sum-ii/)

### 思路
* 典型的**dfs问题**，每个元素都有选或者不选，两种决策。关键是如何去重
    * **可以用类似于枚举排列中有重复元素的情况**，`if(i==cur&&A[i]==A[i-1]) continue`,**cur表示当前考虑的是集合第`cur`个元素**
    * 假设当前考虑到`cur`元素，一共有k个，那么情况就是`选0个，选1个，选2个...选k个`,**这里cur表示，当前考虑的是数组第cur个元素**
### 代码

#### dfs+排列类似去重

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    void dfs(vector<int> &A,int cur,int remain,vector<int> &temp)
    {
            if(remain==0)
            {
                //vis.insert(temp);
                ans.push_back(temp);
                return;
            }

            
        
        //choose or not choose
        
        for(int i = cur;i<A.size();i++)
        {
            if(A[i]>remain) continue;
            if(i!=cur&&A[i]==A[i-1]) continue;
            temp.push_back(A[i]);
            dfs(A,i+1,remain-A[i],temp);
            temp.pop_back();
            
        }
        
    }
    
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<int> temp;
        dfs(candidates,0,target,temp);
        return ans;
    }
};
```
### dfs+直接去重

```cpp
class Solution {
public:
    vector<vector<int>> ans;
    //unordered_set<vector<int>> vis;
    void dfs(vector<int> &A,int cur,int remain,vector<int> &temp)
    {
        if(cur>=A.size())
         {
            if(remain==0)
            {
                ans.push_back(temp);
                return;
            }
        }
        if(cur>=A.size()) return;
            

        
        //choose or not choose
        
         int end = cur+1;
         while(end<A.size()&&A[end-1] == A[end])
             end++;
        
         for(int i=0;cur+i<=end;i++)
         {
             if(remain<A[cur]*i) break;
             for(int k = 1;k<=i;k++)
                 temp.push_back(A[cur]);
             dfs(A,end,remain-A[cur]*i,temp);
             for(int k = 1;k<=i;k++)
                 temp.pop_back();
         }
        

    }
    
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(),candidates.end());
        vector<int> temp;
        dfs(candidates,0,target,temp);
        return ans;
    }
};
```


