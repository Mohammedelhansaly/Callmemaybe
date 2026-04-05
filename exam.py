from typing import List
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if not needle:
            return 0
        for i in range(len(haystack)):
            for j in range(len(needle)):
                if (i + j >= len(haystack)) or haystack[i + j] != needle[j] :
                    break
                else:
                    if j == len(needle) - 1:
                        return i        
        return -1

if __name__ == "__main__":
    solution = Solution()
    nums = [1, 1, 2, 3,3]
    print(solution.strStr("aaa","aaaa"))  # Output: 2
    # print(nums)  # Output: [1, 2, 2]