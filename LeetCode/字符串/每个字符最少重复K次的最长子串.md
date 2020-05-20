# 395. Longest Substring with At Least K Repeating Characters
[题目链接](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/)

### 思路
* `O(n^2)`解法，我们可以枚举子串的长度，然后枚举子串的起点，进行判断，判断有效性可以使用滑动窗口法，可以在`O(1)`的时间内判断出子串是否有效
* `O(n)`解法，使用分治法。满足题意的子串中每个字符至少都出现k次，**则原字符串中出现次数少于k次的字符，一定不会出现在解中**，我们可以依据这些**出现次数少于k次的字符**对原字符串进行划分，在子问题中求解（若字符串中不存在出现次数少于k的字符，则该字符串就是一个解）



### 代码

#### 枚举 + 滑动窗口

滑动窗口一般枚举终点比较好处理

```cpp
class Solution {
public:
    int longestSubstring(string s, int k) {
      
        for(int l = s.length();l>=k;l--)
        {
            int no_greater_k = 0;
            vector<int> ch2num(30,0);
            bool ok = false;
            for(int end = 0;end<s.length();end++)
            {
              if(end>l-1)
               {   if(ch2num[s[end-l]-'a']==1) no_greater_k--;
                  if(ch2num[s[end-l]-'a']==k) no_greater_k++;
                  ch2num[s[end-l]-'a']--; 
               }
              if(ch2num[s[end]-'a']==0)   no_greater_k++;
              if(ch2num[s[end]-'a']==k-1) no_greater_k--;
              ch2num[s[end]-'a']++; 
              if(end>=l-1&&no_greater_k==0){ok = true;break;} 
              
            }
        if(ok) return l;
        }
        return 0;
    }
};
```

#### 分治法

滑动窗口一般枚举终点比较好处理

```cpp
class Solution
{
private:
    int getLongestSubstring(string &s, int begin, int end, int k)
    {
        vector<int> count(26, 0);
        for (int i = begin; i < end; ++i)
        {
            count[s[i] - 'a']++;
        }
        int res = 0;
        for (int i = begin; i < end;)
        {
            //find the first char which appears no less than k
            while (i < end && count[s[i] - 'a'] < k)
                i++;
            if (i == end)
                break;
            int j = i;

            //find the first char which appears less than k
            while (j < end && count[s[j] - 'a'] >= k)
                j++;

            //find an substring which matches the condiation.
            if (i == begin && j == end)
                return end - begin;
            res = max(res, getLongestSubstring(s, i, j, k));
            i = j + 1;
        }
        return res;
    }

public:
    int longestSubstring(string s, int k)
    {
        if (s.size() < k)
            return 0;
        return getLongestSubstring(s, 0, s.size(), k);
    }
};
```