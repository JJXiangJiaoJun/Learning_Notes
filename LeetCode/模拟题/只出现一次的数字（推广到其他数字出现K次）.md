[TOC]
# 137. Single Number II
[题目链接](https://leetcode.com/problems/single-number-ii/)

### 思路
* **因为`int`型变量为32位**,我们用`cnt[i]`表示，**第`i`位为1的个数**
* 因为其他的数字都出现了`K次`，所以我们用`cnt[i] % K`，那么得到的就是只出现一次的数字，在第`i`位的值
* 之后把二进制转化为十进制即可

### 代码

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        vector<int> cnt(32+1,0);
        int i,j,cur_num;
        int len = nums.size();
        for(i = 0;i < len;i++)
        {
            cur_num = nums[i];
            for(int j=31;j>=0;j--){
                cnt[j] += cur_num&1;
                cur_num>>=1;
                if(cur_num==0)
                    break;
            }
            
        }
        
        int ans = 0;
        int K = 3;
        for(i = 0; i < 32;i++)
        {
            int m = cnt[i] % K;
            ans = (ans<<1) + m;
        }
        return ans;
    }
};
```

