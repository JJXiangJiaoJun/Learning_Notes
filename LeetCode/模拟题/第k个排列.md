[TOC]
# 60. Permutation Sequence
[题目链接](https://leetcode.com/problems/permutation-sequence/)

### 思路
* 首先确定是由哪个数字开头`(k/factor[n-i])`
* 然后判断是否后面还有数字
* 重复上述过程，记得每次需要将已经用过的元素删除

### 代码

```cpp
class Solution {
public:
    string getPermutation(int n, int k) {
        string ans;
        vector<int> factor(n+10,1);
        vector<string> num;
        
        for(int i=2;i<=n;i++)
            factor[i] = factor[i-1]*i;
        for(int i=1;i<=n;i++)
            num.push_back(to_string(i));
        
        for(int i=1;i<=n;i++)
        {
            int idx = k / factor[n-i];
            if(k%factor[n-i]!=0)
                idx++;
            ans += num[idx-1];
            num.erase(num.begin() + idx - 1);
            k = k - (idx-1) * factor[n-i];
        }
        
        return ans;
    }
};
```

