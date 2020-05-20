# 347. Top K Frequent Elements
[题目链接](https://leetcode.com/problems/top-k-frequent-elements/)

### 思路
* 使用最小堆来保存前k大的元素
* unordered_map中每个元素遍历时为pair，其中first元素为键，second元素为值。
* priority_queue可以自己定义比较函数,结构体中的operator()，其中greater为升序(小顶堆)、less为降序(大顶堆)

### 代码


```cpp
class Solution {
public:
    typedef pair<int,int> int2;
    
    struct cmp{
        bool operator ()(int2 &a,int2 &b) const
        {
            return a.second > b.second;
        }
    };
    
    
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int> num2cnt;
        
        auto myComp = [](const pair<int, int>& A, const pair<int, int>& B){return A.second < B.second;};
        priority_queue<int2,vector<int2>,cmp> q;
        
        for(int i=0;i<nums.size();i++)
        {
            if(num2cnt.count(nums[i])==0)
                num2cnt[nums[i]] = 1;
            else
                num2cnt[nums[i]]++;
        }
        vector<int> ans;
        
        for(auto &p:num2cnt)
        {
            if(q.size()<k)
                q.push(p);
            else if(p.second>q.top().second)
            {
                q.pop();
                q.push(p);
            }
        }
        
        while(!q.empty())
        {
            ans.push_back(q.top().first);
            q.pop();
        }
        
        return ans;
    }
};
```

