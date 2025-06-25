#include<iostream>
#include<vector>
#include <unordered_map>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <unordered_set>
#include <stack>
#include <string>
using namespace std;

template<typename T>
void printVector(const std::vector<T>& nums)
{
	cout << endl;
	for (const auto& element : nums)
	{
		std::cout << element << " ";
	}
	std::cout << std::endl;
}

template<typename T>
void printVectorOfVector(vector<vector<T>> nums)
{
	cout << endl;
	for (const auto& row:nums)
	{
		for (const auto& element:row)
		{
			cout << element << " ";
		}
		cout << endl;
	}
	cout << endl;
}

struct ListNode {
	int val;
	ListNode* next;
	ListNode():val(0),next(nullptr){}
	ListNode(int x) :val(x), next(nullptr) {}
	ListNode(int x, ListNode* node) :val(x), next(node) {}
};

class Node {
public:
	int val;
	Node* next;
	Node* random;
	Node(int int_val):val(int_val),next(NULL),random(NULL){}
};

struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode():val(0),left(NULL),right(NULL){}
	TreeNode(int val) :val(val), left(NULL), right(NULL) {}
	TreeNode(int val, TreeNode* left, TreeNode* right) :val(val), left(left), right(right) {}
};

void printListNode(ListNode* head)
{
	ListNode* p = head;
	while (p != NULL)
	{
		cout << p->val << " ";
		p = p->next;
	}
	cout << endl;
}

ListNode* VectorToListNode(vector<int> nums)
{
	ListNode* dummynode = new ListNode(0);
	ListNode* p = dummynode;
	for (int i = 0; i < nums.size(); i++)
	{
		p->next = p->next = new ListNode(nums[i]);
		p = p->next;
	}
	return dummynode->next;
}

vector<int> twosum(vector<int> nums, int target)
{	
	unordered_map<int, int> map;
	for (int i = 0; i < nums.size(); i++)
	{
		int find = target - nums[i];
		if (map.count(find))
			return { map[find], i };
		map[nums[i]] = i;
	}
	return {};
}

void Test_twosum()
{
	int target;
	cout << "hot100 1.两数之和" << endl;
	cout << "\ninput the val of target at first:" << endl;
	cin >> target;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> res = twosum(nums, target);
	cout << "\nthe result of twosum (array index):" << endl;
	printVector(res);
}

vector<vector<string>> groupAnagrams(vector<string>& strs)
{
	vector<vector<string>> res;
	unordered_map<string, vector<string>> map;
	for (int i = 0; i < strs.size(); i++)
	{
		string str = strs[i];
		sort(str.begin(), str.end());
		map[str].push_back(strs[i]);
	}
	for (const pair<string, vector<string>>& temp : map)
		res.push_back(temp.second);
	return res;
}

void Test_groupAnagrams()
{
	string str;
	vector<string> strs;
	cout << "hot100 49.字母异位词分组" << endl;
	cout << "\ninput the array of string:" << endl;
	while (cin >> str)
	{
		strs.push_back(str);
		if (cin.peek() == '\n')
			break;
	}
	vector<vector<string>> res = groupAnagrams(strs);
	cout << "\nthe res of groupAnagrams:" << endl;
	printVectorOfVector(res);
}

int longestConsecutive(vector<int>& nums)
{
	if (nums.empty()) return 0;
	int res = 1;
	unordered_set<int> myset(nums.begin(),nums.end());
	for (int num:myset)//set自动去重，遍历nums会遍历有重复的元素
	{
		myset.insert(num);
		int temp = 1;
		int fnum= num+1;
		while (myset.count(fnum))
		{
			fnum++;
			temp++;
		}
		res = max(res, temp);
	}
	return res;
}

void Test_longestConsecutive()
{
	vector<int> nums;
	int num;
	cout << "hot100 128.最长连续序列" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = longestConsecutive(nums);
	cout << "\nthe res is: " << res << endl;
}

void moveZeroes(vector<int>& nums)
{
	int left = 0;
	int right = 0;
	while (right < nums.size()) {
		if (nums[right] != 0)
		{
			nums[left] = nums[right];
			left++;
		}
		right++;
	}
	while (left < nums.size()) {
		nums[left] = 0;
		left++;
	}
}

void Test_moveZeroes()
{
	vector<int> nums;
	int num;
	cout << "hot100 283.移动零" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	moveZeroes(nums);
	printVector(nums);
}

int maxArea(vector<int>& height)
{
	int left = 0, right = height.size()-1;
	int res = INT_MIN;
	while (left < right)
	{
		int temp = (right - left) * min(height[left], height[right]);
		res = max(res, temp);
		if (height[left] <= height[right])
		{left++;}
		else { right--;}
	}
	return res;
}

void Test_maxArea()
{
	vector<int> nums;
	int num;
	cout << "hot100 11.盛最多水的容器" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = maxArea(nums);
	cout << "\nthe res is :" << res << endl;
}

vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> res;
	sort(nums.begin(), nums.end());
	for (int i = 0; i < nums.size(); i++) {
		if (i - 1 >= 0 && nums[i] == nums[i - 1])
			continue;
		int target = -nums[i];
		int left = i + 1, right = nums.size() - 1;
		while (left < right) {
			int sum = nums[left] + nums[right];
			if (sum == target) {
				res.push_back({ nums[i], nums[left], nums[right] });
				//如果这个和前一个相等，就继续往前跳, 而且一定要left<right
				while (left < right && nums[left] == nums[left + 1]) { left++; }
				while (left < right && nums[right] == nums[right - 1]) { right--; }
				left++;
				right--;
			}
			else if (sum > target)
				right--;
			else if (sum < target)
				left++;
		}
	}
	return res;
}

void Test_threeSum()
{
	cout << " hot100 15.三数之和 " << endl;
	cout << "\ninput the num of array:" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<vector<int>> res = threeSum(nums);
	printVectorOfVector(res);
}

int trap(vector<int>& height)
{
	int n = height.size();
	vector<int> leftmax(n, 0);
	vector<int> rightmax(n, 0);
	int maxval = 0;
	for (int i = 0; i < n; i++)
	{
		maxval = max(maxval, height[i]);
		leftmax[i] = maxval;
	}
	maxval = 0;
	for (int i = n-1; i >=0 ; i--)
	{
		maxval = max(maxval, height[i]);
		rightmax[i] = maxval;
	}
	int res = 0;
	for (int i = 1; i < n - 1; i++)
	{
		res += (min(leftmax[i], rightmax[i]) - height[i]);
	}
	return res;
}

void Test_trap()
{
	cout << " hot100 42.接雨水 " << endl;
	cout << "\ninput the num of array:" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = trap(nums);
	cout << "\nThe res is:"<<res << endl;
}

int lengthOfLongestSubstring(string s) {
	if (s.empty()) return 0;
	unordered_map<char, int> map;
	int slow = 0, fast = 0;
	int res = 0;
	while (fast < s.size())
	{
		char qian = s[fast];
		map[qian]++;
		fast++;
		while (map[qian] > 1) {
			char hou = s[slow];
			map[hou] = (map[hou] >= 1 ? (map[hou] - 1) : 0);
			slow++;
		}
		res = max(res, (fast - slow));
	}
	return res;
}

void Test_lengthOfLongestSubstring()
{
	cout << " hot100 3.无重复字符的最长子串 " << endl;
	cout << "\ninput string:" << endl;
	string str;
	cin >> str;
	int res = lengthOfLongestSubstring(str);
	cout << "\nThe res is:" << res << endl;
}

vector<int> findAnagrams(string s, string p)
{
	unordered_map<char, int> dict;
	for (char c : p)
		dict[c]++;
	int size = p.size();
	int left = 0, right = 0;
	unordered_map<char, int> window;
	int valid = 0;
	vector<int> res;
	while (right < s.size())
	{
		char c = s[right];
		if (dict.count(c))
		{
			window[c]++;
			if (dict[c] == window[c])
				valid++;
		}
		right++;
		while (right - left >= size)
		{
			if (valid == dict.size())
				res.push_back(left);
			char b = s[left];
			if (dict.count(b))
			{
				if (dict[b] == window[b])
					valid--;
				window[b]--;
			}
			left++;
		}
	}
	return res;
}

void Test_findAnagrams()
{
	string s;
	string p; 
	cout << "438. 找到字符串中所有字母异位词" << endl;
	cout << "\n输入长字符串：" << endl;
	cin >> s;
	cout << "\n输入异位词：" << endl;
	cin >> p;
	vector<int> res = findAnagrams(s, p);
	cout << "\n计算的结果是：" << endl;
	printVector(res);
}

int subarraySum(vector<int>& nums, int k)
{
	unordered_map<int, int> map;
	int res = 0;
	map[0] = 1;
	int sum = 0;
	for (int i = 0; i < nums.size(); i++)
	{
		sum += nums[i];
		int target = sum - k;
		if (map.count(target))
		{
			res += map[target];
		}
		map[sum]++;
	}
	return res;
}

void Test_subarraySum()
{
	cout << "560. 和为 K 的子数组" << endl;
	cout << "\n输入数组nums：" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\n输入target：" << endl;
	cin >> target;
	
	int res = subarraySum(nums, target);
	cout << "\n计算的结果是："<< res << endl;
}

class Myqueue {
public:
	deque<int> deq;

	void push(int num)
	{
		while (!deq.empty() && num > deq.back())
			deq.pop_back();
		deq.push_back(num);
	}

	void pop(int num)
	{
		if (!deq.empty() && num == deq.front())
			deq.pop_front();
	}

	int getval()
	{
		return deq.front();
	}
};

vector<int> maxSlidingWindow(vector<int>& nums, int k) {
	Myqueue que;
	vector<int> res;
	for (int i = 0; i < k; i++)
	{
		que.push(nums[i]);
	}
	res.push_back(que.getval());
	for (int i = k; i < nums.size(); i++)
	{
		que.pop(nums[i - k]);
		que.push(nums[i]);
		res.push_back(que.getval());
	}
	return res;
}

void Test_maxSlidingWindow()
{
	cout << "239. 滑动窗口最大值" << endl;
	cout << "\n输入数组nums：" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\n输入target：" << endl;
	cin >> target;

	vector<int> res = maxSlidingWindow(nums, target);
	cout << "\n计算的结果是：" <<endl;
	printVector(res);
}

string minWindow(string s, string t) {
	unordered_map<char, int> map, window;
	for (int i = 0; i < t.size(); i++)
	{map[t[i]]++;}
	int left = 0, right = 0;
	int start = 0, length = INT_MAX;
	int valid = 0;
	while (right < s.size())
	{
		char b = s[right];
		right++;
		if (map.count(b))
		{
			window[b]++;
			if (map[b] == window[b])
				valid++;
		}
		while (valid == map.size())
		{
			if (right - left < length)
			{
				length = right - left;
				start = left;
			}
			char c = s[left];
			if (map.count(c))
			{
				if (map[c] == window[c])
					valid--;
				window[c]--;
			}
			left++;
		}
	}
	return length == INT_MAX ? "" : s.substr(start, length);
}

void Test_minWindow()
{
	string s;
	string p;
	cout << "76. 最小覆盖子串" << endl;
	cout << "\n输入长字符串：" << endl;
	cin >> s;
	cout << "\n输入短字符串：" << endl;
	cin >> p;
	string res = minWindow(s, p);
	cout << "\n计算的结果是：" << endl;
	cout << res << endl;
}

int maxSubArray(vector<int>& nums) {
#if 1
	cout << "\n动态规划：" << endl;
	vector<int> dp(nums.size());
	dp[0] = nums[0];
	int res = dp[0];
	for (int i = 1; i < nums.size(); i++)
	{
		dp[i] = max(dp[i - 1] + nums[i], nums[i]);
		res = max(res, dp[i]);
	}
	return res;
#else
	cout << "\n贪心：" << endl;
	int sum = 0;
	int res = INT_MIN;
	for (int i = 0; i < nums.size(); i++)
	{
		sum += nums[i];
		res = max(res, sum);
		if (sum < 0)
			sum = 0;
	}
	return res;
#endif 

	
}

void Test_maxSubArray()
{
	cout << "53. 最大子数组和" << endl;
	cout << "\n输入数组nums：" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = maxSubArray(nums);
	cout << "\n计算的结果是：" <<res<< endl;
}

vector<vector<int>> merge(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end(), [](const vector<int>& a, const vector<int>& b) {
		return a[0] < b[0];
		});
	vector<vector<int>> res;
	int left = intervals[0][0], right = intervals[0][1];
	for (int i = 1; i < intervals.size(); i++)
	{
		if (right >= intervals[i][0] && right <= intervals[i][1])
		{
			right = intervals[i][1];
		}
		else if (right < intervals[i][0])
		{
			res.push_back({ left,right });
			left = intervals[i][0];
			right = intervals[i][1];
		}
	}res.push_back({ left,right });
	return res;
}

void Test_merge() {
	cout << "56. 合并区间" << endl;
	cout << "\n依次输入区间（a, b）：" << endl;
	cout << "输入非数字字符结束输入" << endl;
	vector<vector<int>> nums;
	int a, b;
	while (cin >> a >> b)
	{
		nums.push_back({a, b});
	}
	vector<vector<int>> res = merge(nums);
	cout << "\n计算的结果是：" << endl;
	printVectorOfVector(res);
}

void reverse_nums(vector<int>& nums, int begin, int end)
{
	while (begin <= end)
	{
		int temp = nums[begin];
		nums[begin] = nums[end];
		nums[end] = temp;
		begin++;
		end--;
	}
}

void rotate(vector<int>& nums, int k) {
	int n = nums.size();
	k = k % n;
	reverse_nums(nums, 0, n - 1);
	reverse_nums(nums, 0, k - 1);
	reverse_nums(nums, k, n - 1);
}

void Test_rotate()
{
	cout << "189. 轮转数组" << endl;
	cout << "\n依次输入数组元素：" << endl;
	vector<int> nums;
	int num;
	int k;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	cout << "\n输入向右轮转的位置k：" << endl;
	cin >> k;
	rotate(nums, k);
	cout << "\n计算的结果是：" << endl;
	printVector(nums);
}

vector<int> productExceptSelf(vector<int>& nums) {
	int n = nums.size();
	vector<int> left(n, nums[0]);
	vector<int> right(n, nums[n - 1]);
	for (int i = 1; i < nums.size(); i++)
	{
		left[i] = left[i - 1] * nums[i];
	}
	for (int i = n - 2; i >= 0; i--)
	{
		right[i] = right[i + 1] * nums[i];
	}
	vector<int> res(n, 0);
	res[0] = right[1];
	res[n - 1] = left[n - 2];
	for (int i = 1; i < nums.size() - 1; i++)
	{
		res[i] = left[i - 1] * right[i + 1];
	}
	return res;
}

void Test_productExceptSelf()
{
	cout << "238. 除自身以外数组的乘积" << endl;
	cout << "\n依次输入数组元素：" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> res = productExceptSelf(nums);
	cout << "\n计算的结果是：" << endl;
	printVector(res);
}

int firstMissingPositive(vector<int>& nums) {
	int n = nums.size();
	for (int i = 0; i < n; i++)
	{
		while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1])
		{
			swap(nums[i], nums[nums[i] - 1]);
		}
	}
	for (int i = 0; i < n; i++)
	{
		if (nums[i] != i+1)
			return i + 1;
	}
	return n + 1;
}

void Test_firstMissingPositive() {
	cout << "41. 缺失的第一个正数" << endl;
	cout << "\n依次输入数组元素：" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = firstMissingPositive(nums);
	cout << "\n计算的结果是：" << res << endl;
}

void setZeroes(vector<vector<int>>& matrix) {
	int m = matrix.size();
	int n = matrix[0].size();
	vector<bool> row(m, false), col(n, false);
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (matrix[i][j] == 0)
			{
				row[i] = true;
				col[j] = true;
			}
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (row[i]||col[j])
			{
				matrix[i][j] = 0;
			}
		}
	}
}

void Test_setZeroes()
{
	cout << "73. 矩阵置零" << endl;
	int m, n;
	cout << "\n输入矩阵的行数m和列数n：" << endl;
	cin >> m >> n;
	cout << "从左上到右下依次输入矩阵元素" << endl;
	vector<vector<int>> nums(m, vector<int>(n));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	setZeroes(nums);
	cout << "\n计算的结果是：" << endl;
	printVectorOfVector(nums);
}

vector<int> spiralOrder(vector<vector<int>>& matrix)
{
	vector<int> res;
	int left = 0, right = matrix[0].size() - 1;
	int top = 0, bottom = matrix.size() - 1;
	while (left <= right && top <= bottom) {
		for (int j = left; j <= right; j++) {
			res.push_back(matrix[top][j]);
		}
		top++;

		for (int i = top; i <= bottom; i++) {
			res.push_back(matrix[i][right]);
		}
		right--;

		if (top <= bottom) {
			for (int j = right; j >= left; j--) {
				res.push_back(matrix[bottom][j]);
			}
			bottom--;
		}

		if (left <= right) {
			for (int i = bottom; i >= top; i--) {
				res.push_back(matrix[i][left]);
			}
			left++;
		}
	}
	return res;
}

void Test_spiralOrder()
{
	int m, n;
	cout << "\n请输入矩阵的行数m和列数n:" << endl;
	cin >> m >> n;
	cout << "\n依次从左上角到右下角输入矩阵元素" << endl;
	vector<vector<int>> nums(m, vector<int>(n,0));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	vector<int> res = spiralOrder(nums);
	printVector(res);
}

void rotate_nums(vector<vector<int>>& matrix) {
	int m = matrix.size();
	for (int i = 0; i < m; i++)
	{
		for (int j = i; j < m; j++)
		{
			int temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m / 2; j++)
		{
			int temp = matrix[i][m-j-1];
			matrix[i][m-j-1] = matrix[i][j];
			matrix[i][j] = temp;
		}
	}
}

void Test_rotate_nums()
{
	int n;
	cout << "\n请输入n*n矩阵的大小n:" << endl;
	cin >> n;
	cout << "\n依次从左上角到右下角输入矩阵元素" << endl;
	vector<vector<int>> nums(n, vector<int>(n, 0));
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	rotate_nums(nums);
	printVectorOfVector(nums);
}

bool searchMatrix(vector<vector<int>>& matrix, int target) {
	int m = matrix.size() - 1;
	int n = matrix[0].size() - 1;
	int starx = 0, stary = n;
	while (starx <= m && stary >= 0)
	{
		if (matrix[starx][stary] > target)
		{
			stary--;
		}
		else if (matrix[starx][stary] < target)
		{
			starx++;
		}
		else {
			return true;
		}
	}
	return false;
}

void Test_searchMatrix()
{
	int m, n,target;
	cout << "\n请输入矩阵的行数m和列数n:" << endl;
	cin >> m >> n;
	cout << "\n依次从左上角到右下角输入矩阵元素" << endl;
	vector<vector<int>> nums(m, vector<int>(n, 0));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	cout << "\n请输入目标值target:" << endl;
	cin >> target;

	bool res = searchMatrix(nums, target);
	cout << res << endl;
}

ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
	ListNode* p1 = headA;
	ListNode* p2 = headB;
	while (p1 != p2)
	{
		p1 = (p1 == nullptr) ? headB : p1->next;
		p2 = (p2 == nullptr) ? headA : p2->next;
		if (p1 == nullptr && p2 == nullptr) return nullptr;
	}
	return p1;
}

void Test_getIntersectionNode()
{
	ListNode a1(4); 
	ListNode a2(1); a1.next = &a2;
	ListNode b1(5);
	ListNode b2(6); b1.next = &b2;
	ListNode b3(1); b2.next = &b3;
	ListNode c1(8); a2.next = &c1; b3.next = &c1;
	ListNode c2(4); c1.next = &c2;
	ListNode c3(5); c2.next = &c3;
	ListNode* headA = &a1;
	ListNode* headB = &b1;
	ListNode* res = getIntersectionNode(headA, headB);
	if (res == nullptr)
		cout << "null" << endl;
	else
		cout << res->val << endl;
}

ListNode* reverseList(ListNode* head) {
	ListNode* p = head;
	ListNode* pre = NULL;
	while (p != NULL)
	{
		ListNode* temp = p->next;
		p->next = pre;
		pre = p;
		p = temp;
	}
	return pre;
}

void Test_reverseList()
{
	ListNode a1(1);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(3); a2.next = &a3;
	ListNode a4(4); a3.next = &a4;
	ListNode a5(5); a4.next = &a5;
	ListNode* head = &a1;
	ListNode* res = reverseList(head);
	printListNode(res);
}

bool isPalindrome(ListNode* head) {
	stack<int> sta;
	ListNode* p1 = head;
	while (p1 != nullptr)
	{
		sta.push(p1->val);
		p1 = p1->next;
	}
	ListNode* p2 = head;
	while (p2 != NULL)
	{
		int temp = sta.top();
		sta.pop();
		if(p2->val!=temp)
		{
			return false;
		}
		p2 = p2->next;
	}
	return true;
}

void Test_isPalindrome()
{
	ListNode a1(1);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(2); a2.next = &a3;
	ListNode a4(3); a3.next = &a4;
	ListNode* head = &a1;
	bool res = isPalindrome(head);
	string resout = (res == true) ? "TRUE" : "FALSE";
	cout << resout << endl;
}

bool hasCycle(ListNode* head) {
	ListNode* p1 = head;
	ListNode* p2 = head;
	while (p1 != NULL && p1->next != NULL && p2 != NULL) {
		p1 = p1->next->next;
		p2 = p2->next;
		if (p1 == p2) {
			return true;
		}
	}
	return false;
}

void Test_hasCycle()
{
	ListNode a1(3);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(0); a2.next = &a3;
	ListNode a4(4); a3.next = &a4; a4.next = &a2;
	ListNode* head = &a1;
	bool res = hasCycle(head);
	string resout = (res == true) ? "TRUE" : "FALSE";
	cout << resout << endl;
}

ListNode* detectCycle(ListNode* head) {
	ListNode* p1 = head;
	ListNode* p2 = head;
	while (p1 != NULL && p1->next != NULL && p2 != NULL) {
		p1 = p1->next->next;
		p2 = p2->next;
		if (p1 == p2) {
			ListNode* p3 = head;
			while (p3 != p2)
			{
				p3 = p3->next;
				p2 = p2->next;
			}
			return p3;
		}
	}
	return NULL;
}

void Test_detectCycle()
{
	ListNode a1(3);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(0); a2.next = &a3;
	ListNode a4(4); a3.next = &a4; a4.next = &a2;
	ListNode* head = &a1;
	ListNode* res = detectCycle(head);
	if (res == NULL)
		cout << "NULL" << endl;
	int resval = res->val;
	cout << resval<< endl;
}

ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
	ListNode dummynode(0);
	ListNode* p = &dummynode;
	while (list1 != NULL && list2 != NULL) {
		if (list1->val > list2->val) {
			ListNode* newnode = new ListNode(list2->val);
			p->next = newnode;
			p = p->next;
			list2 = list2->next;
		}
		else {
			ListNode* newnode = new ListNode(list1->val);
			p->next = newnode;
			p = p->next;
			list1 = list1->next;
		}
	}
	if (list1 != NULL) {
		p->next = list1;
	}
	if (list2 != NULL) {
		p->next = list2;
	}
	return dummynode.next;
}

void Test_mergeTwoLists()
{
	ListNode a1(1);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(4); a2.next = &a3;

	ListNode b1(1);
	ListNode b2(3); b1.next = &b2;
	ListNode b3(4); b2.next = &b3;

	ListNode* heada = &a1;
	ListNode* headb = &b1;
	ListNode* res = mergeTwoLists(heada, headb);
	printListNode(res);
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	ListNode* dummynode = new ListNode(0);
	ListNode* p = dummynode;
	int lastval = 0;
	while (l1 != NULL && l2 != NULL) {
		int sum = l1->val + l2->val + lastval;
		lastval = sum / 10;
		sum = sum % 10;
		ListNode* newnode = new ListNode(sum);
		p->next = newnode;
		l1 = l1->next;
		l2 = l2->next;
		p = p->next;
	}
	while (l1 != NULL) {
		int sum = l1->val + lastval;
		lastval = sum / 10;
		sum = sum % 10;
		ListNode* newnode = new ListNode(sum);
		p->next = newnode;
		l1 = l1->next;
		p = p->next;
	}
	while (l2 != NULL) {
		int sum = l2->val + lastval;
		lastval = sum / 10;
		sum = sum % 10;
		ListNode* newnode = new ListNode(sum);
		p->next = newnode;
		l2 = l2->next;
		p = p->next;
	}
	if (lastval != 0)
	{
		ListNode* newnode = new ListNode(lastval);
		p->next = newnode;
	}
	return dummynode->next;
}

void Test_addTwoNumbers()
{
	ListNode a1(2);
	ListNode a2(4); a1.next = &a2;
	ListNode a3(3); a2.next = &a3;

	ListNode b1(5);
	ListNode b2(6); b1.next = &b2;
	ListNode b3(4); b2.next = &b3;

	ListNode* heada = &a1;
	ListNode* headb = &b1;
	ListNode* res = addTwoNumbers(heada, headb);
	printListNode(res);
}

ListNode* removeNthFromEnd(ListNode* head, int n) {
	ListNode* dummynode = new ListNode(0);
	dummynode->next = head;
	ListNode* p1 = dummynode;
	ListNode* p2 = dummynode;
	while (n--)
	{
		p1 = p1->next;
	}
	p1 = p1 -> next;
	while (p1 != NULL)
	{
		p1 = p1->next;
		p2 = p2->next;
	}
	ListNode* temp = p2->next;
	p2->next = p2->next->next;
	delete temp;
	return dummynode->next;
}

void Test_removeNthFromEnd()
{
	cout << " 19. 删除链表的倒数第 N 个结点 " << endl;
	vector<int> nums;
	int num;
	cout << " 依次输入链表元素 " << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << " 输入删去倒数第几个元素 " << endl;
	cin >> target;
	ListNode* head = VectorToListNode(nums);
	ListNode* res = removeNthFromEnd(head, target);
	printListNode(res);
}

ListNode* swapPairs(ListNode* head) {
	if (head == NULL || head->next == NULL)
		return head;
	ListNode* slow = head;
	ListNode* fast = head->next;
	ListNode* temp = fast->next;
	fast->next = slow;
	slow->next = swapPairs(temp);
	return fast;
}

void Test_swapPairs() {
	ListNode a1(1);
	ListNode a2(2); a1.next = &a2;
	ListNode a3(3); a2.next = &a3;
	ListNode a4(4); a3.next = &a4;
	ListNode* head = &a1;
	ListNode* res = swapPairs(head);
	printListNode(res);
}

ListNode* reverse(ListNode* left, ListNode* right)
{
	ListNode* p = left;
	ListNode* pre = nullptr;
	while (p != right)
	{
		ListNode* temp = p->next;
		p->next = pre;
		pre = p;
		p = temp;
	}
	return pre;
}

ListNode* reverseKGroup(ListNode* head, int k) {
	ListNode* left = head;
	ListNode* right = head;
	for (int i = k; i > 0; i--)
	{
		if (right == nullptr)
			return head;
		right = right->next;
	}
	ListNode* node = reverse(left, right);
	left->next = reverseKGroup(right, k);
	return node;
}

void Test_reverseKGroup()
{
	cout << " 25. K 个一组翻转链表 " << endl;
	vector<int> nums;
	int num;
	cout << "依次输入链表元素" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "输入几个一组：" << endl;
	cin >> target;
	ListNode* head = VectorToListNode(nums);
	ListNode* res = reverseKGroup(head, target);
	printListNode(res);
}

Node* copyRandomList(Node* head) {
	unordered_map<Node*, Node*> map;
	for (Node* p = head; p != NULL; p = p->next)
	{
		if (map.find(p) == map.end())
			map[p] = new Node(p->val);
	}
	for (Node* p = head; p != NULL; p = p->next)
	{
		if (p->next != NULL)
			map[p]->next = map[p->next];
		if (p->random != NULL)
			map[p]->random = map[p->random];
	}
	return map[head];
}

void printNode(Node* head)
{
	for (Node* p = head; p != NULL; p = p->next)
	{
		cout << "Node: " << p->val;
		cout << ", Next: ";
		if (p->next) cout << p->next->val; else cout << "null";
		cout << ", Random: ";
		if (p->random) cout << p->random->val; else cout << "null";
		cout << endl;
	}
}

void Test_copyRandomList()
{
	Node a1(7);
	Node a2(13);
	Node a3(11);
	Node a4(10);
	Node a5(1);
	a1.next = &a2; a2.next = &a3; a3.next = &a4; a4.next = &a5; a5.next = nullptr;
	a1.random = nullptr; a2.random = &a1; a3.random = &a5; a4.random = &a3; a5.random = &a1;
	Node* head = &a1;
	Node* res = copyRandomList(head);
	printNode(res);
}

ListNode* sortList(ListNode* head) {
	if (head == NULL || head->next == NULL)
		return head;
	ListNode* slow = head;
	ListNode* fast = head->next;
	while (fast != NULL&&fast->next!=NULL)
	{
		fast = fast->next->next;
		slow = slow->next;
	}
	ListNode* newhead = slow->next;
	slow->next = nullptr;
	ListNode* left = sortList(head);
	ListNode* right = sortList(newhead);

	ListNode* dummynode = new ListNode(0);
	ListNode* p = dummynode;
	while (left != nullptr && right != nullptr)
	{
		if (left->val < right->val)
		{
			p->next = left;
			left = left->next;
		}
		else {
			p->next = right;
			right = right->next;
		}
		p = p->next;
	}
	if (left != nullptr)
		p->next = left;
	if (right != nullptr)
		p->next = right;
	return dummynode->next;
}

void Test_sortList()
{
	cout << " 148. 排序链表 " << endl;
	vector<int> nums;
	int num;
	cout << "依次输入链表元素" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	ListNode* head = VectorToListNode(nums);
	ListNode* res = sortList(head);
	printListNode(res);
}

ListNode* inputLists()
{
	vector<int> nums;
	int num;
	cout << "\n依次输入链表元素" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	ListNode* head = VectorToListNode(nums);
	return head;
}

ListNode* mergeKLists(vector<ListNode*>& lists) {
	auto myfunc = [](ListNode* a, ListNode* b) {return a->val > b->val; };
	priority_queue < ListNode*, vector<ListNode*>, decltype(myfunc)> que(myfunc);
	for (int i = 0; i < lists.size(); i++)
	{
		if(lists[i]!=nullptr)
		que.push(lists[i]);
	}
	ListNode* dummynode = new ListNode(0);
	ListNode* p = dummynode;
	while (!que.empty()) {
		ListNode* node = que.top();
		que.pop();
		if (node->next != nullptr)
		{
			que.push(node->next);
		}
		p->next = node;
		p = p->next;
	}
	return dummynode->next;
}

void Test_mergeKLists()
{
	cout << " 23. 合并 K 个升序链表 " << endl;
	ListNode* head1 = inputLists();
	ListNode* head2 = inputLists();
	ListNode* head3 = inputLists();
	vector<ListNode*> lists = {head1, head2, head3};
	ListNode* res = mergeKLists(lists);
	cout << endl;
	printListNode(res);
}

class LRUCache {
public:
	LRUCache(int capacity) {
		cap = capacity;
	}

	int get(int key) {
		if (mapcache.find(key) == mapcache.end())
		{
			return -1;
		}
		else {
			pair<int, int> node = *mapcache[key];
			cachelist.erase(mapcache[key]);
			cachelist.push_front(node);
			mapcache[key] = cachelist.begin();
			return node.second;
		}
	}

	void put(int key, int value) {
		if (mapcache.find(key) == mapcache.end())
		{
			if (cachelist.size() >= cap)
			{
				mapcache.erase(cachelist.back().first);
				cachelist.pop_back();
			}
			cachelist.push_front({ key,value});
			mapcache[key] = cachelist.begin();
		}
		else {
			cachelist.erase(mapcache[key]);
			cachelist.push_front({ key,value });
			mapcache[key] = cachelist.begin();
		}
	}
private:
	list<pair<int, int>> cachelist;
	unordered_map<int, list<pair<int, int>>::iterator> mapcache;
	int cap;
};

void Test_LRUCache()
{
	LRUCache mylru(2);
	mylru.put(1, 1);
	mylru.put(2, 2);
	cout << mylru.get(1) << endl;//1
	mylru.put(3, 3);
	cout << mylru.get(2) << endl;//-1
	mylru.put(4, 4);
	cout << mylru.get(1) << endl;//-1
}

void inordertraversal(TreeNode* root, vector<int>& res)
{
	if (root == NULL)
		return;
	inordertraversal(root->left, res);
	res.push_back(root->val);
	inordertraversal(root->right, res);
}

vector<int> inorderTraversal(TreeNode* root) {
	vector<int> res;
	inordertraversal(root, res);
	return res;
}

void Test_inorderTraversal()
{
	cout << "hot100 94.  二叉树的中序遍历" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	vector<int> res = inorderTraversal(root);
	printVector(res);
}

int maxDepth(TreeNode* root) {
	if (root == nullptr)
		return 0;
	int left = maxDepth(root->left) + 1;
	int right = maxDepth(root->right) + 1;
	return max(left, right);
}

void Test_maxDepth()
{
	cout << "hot100 104. 二叉树的最大深度" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	int res = maxDepth(root);
	cout << "the max depth is " << res << endl;
}

TreeNode* invertTree(TreeNode* root) {
	if (root == nullptr)
		return root;
	TreeNode* left = invertTree(root->left);
	TreeNode* right = invertTree(root->right);
	TreeNode* temp = left;
	root->left = right;
	root->right = temp;
	return root;
}

void Test_invertTree()
{
	cout << "hot100 226. 翻转二叉树" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	TreeNode* res = invertTree(root);
	vector<int> resv = inorderTraversal(res);
	printVector(resv);
}

bool issymmetric(TreeNode* left, TreeNode* right)
{
	if (left == nullptr && right == nullptr)
		return true;
	if (left == nullptr || right == nullptr)
		return false;
	else if (left->val != right->val)
		return false;
	bool leftB = issymmetric(left->left, right->right);
	bool rightB = issymmetric(left->right, right->left);
	return leftB && rightB;

}
bool isSymmetric(TreeNode* root) {
	return issymmetric(root->left, root->right);
}

void Test_isSymmetric()
{
	cout << "hot100 101. 对称二叉树" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	bool res = isSymmetric(root);
	string resstr = (res == true) ? "TRUE" : "FALSE";
	cout << resstr << endl;
}

int findValue(TreeNode* root, int& maxres)
{
	if (root == nullptr)
		return 0;
	int leftV = findValue(root->left, maxres);
	int rightV = findValue(root->right, maxres);
	maxres = max(maxres, (leftV + rightV));
	return max(leftV, rightV) + 1;
}

int diameterOfBinaryTree(TreeNode* root) {
	int maxres = 0;
	findValue(root, maxres);
	return maxres;
}

void Test_diameterOfBinaryTree()
{
	cout << "hot100 543. 二叉树的直径" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	int res = diameterOfBinaryTree(root);
	cout << res << endl;
}

vector<vector<int>> levelOrder(TreeNode* root) {
	vector<vector<int>> res;
	if (root == nullptr) return res;
	queue<TreeNode*> que;
	que.push(root);
	while (!que.empty())
	{
		int size = que.size();
		vector<int> temp;
		for (int i = 0; i < size; i++)
		{
			TreeNode* node = que.front();
			que.pop();
			temp.push_back(node->val);
			if (node->left) que.push(node->left);
			if (node->right) que.push(node->right);
		}
		res.push_back(temp);
	}
	return res;
}

void Test_levelOrder()
{
	cout << "hot100 102. 二叉树的层序遍历" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	vector<vector<int>> res = levelOrder(root);
	printVectorOfVector(res);
}

TreeNode* sortedarraytoBST(vector<int>& nums, int begin, int end)
{
	if (begin > end)
		return nullptr;
	int index = (begin + end) / 2;
	TreeNode* root = new TreeNode(nums[index]);
	root->left = sortedarraytoBST(nums, begin, index - 1);
	root->right = sortedarraytoBST(nums, index + 1, end);
	return root;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {
	return sortedarraytoBST(nums, 0, nums.size() - 1);
}

void Test_sortedArrayToBST()
{
	cout << "hot100 108. 将有序数组转换为二叉搜索树" << endl;
	vector<int> nums;
	int num;
	cout << "\n依次输入升序数组的元素" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	TreeNode* res = sortedArrayToBST(nums);
	vector<int> resvectror= inorderTraversal(res);
	printVector(resvectror);
}

TreeNode* isValidBST_pre = nullptr;
bool isValidBST(TreeNode* root) {
	if (root == nullptr)
		return true;
	bool left = isValidBST(root->left);
	if (isValidBST_pre != nullptr)
	{
		if (isValidBST_pre->val >= root->val)
			return false;
	}
	isValidBST_pre = root;
	bool right = isValidBST(root->right);
	return left && right;
}


void Test_isValidBST()
{
	cout << "hot100 98. 验证二叉搜索树" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	bool res = isValidBST(root);
	string resstr = (res == true) ? "true" : "false";
	cout << resstr << endl;
}

int findkthSmallest_res;
void findkthSmallest(TreeNode* root, int& k)
{
	if (root == nullptr)
		return;
	findkthSmallest(root->left, k);
	k--;
	if (k == 0)
	{
		findkthSmallest_res = root->val;
		return;
	}
	findkthSmallest(root->right, k);
}

int kthSmallest(TreeNode* root, int k) {
	findkthSmallest(root, k);
	return findkthSmallest_res;
}

void Test_kthSmallest()
{
	cout << "hot100 230. 二叉搜索树中第 K 小的元素" << endl;
	vector<int> nums;
	int num;
	cout << "\n依次输入升序数组的元素" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int k;
	cout << "\n输入需要查找第k小的元素:" << endl;
	cin >> k;
	TreeNode* nodeTree = sortedArrayToBST(nums);
	int res = kthSmallest(nodeTree, k);
	cout << "\nres：" << res << endl;
}

vector<int> rightSideView(TreeNode* root) {
	vector<int> res;
	queue<TreeNode*> que;
	if (root == nullptr)
		return res;
	que.push(root);
	while (!que.empty())
	{
		int size = que.size();
		for (int i = 0; i < size; i++)
		{
			TreeNode* node = que.front();
			que.pop();
			if (i == size - 1)
				res.push_back(node->val);
			if (node->left) que.push(node->left);
			if (node->right) que.push(node->right);
		}
	}
	return res;
}

void Test_rightSideView()
{
	cout << "hot100 199. 二叉树的右视图" << endl;
	TreeNode* root = new TreeNode(0);
	root->left = new TreeNode(1);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(2);
	vector<int> res = rightSideView(root);
	printVector(res);
}

void flatten(TreeNode* root) {
	if (root == nullptr)
		return;
	flatten(root->left);
	flatten(root->right);
	TreeNode* left = root->left;
	TreeNode* right = root->right;
	root->right = left;
	root->left = nullptr;
	TreeNode* p = root;
	while (p->right != nullptr)
		p = p->right;
	p->right = right;
	return;
}

void Test_flatten()
{
	cout << "hot100 114. 二叉树展开为链表" << endl;
	TreeNode* root = new TreeNode(1);
	root->left = new TreeNode(2);
	root->left->left = new TreeNode(3);
	root->left->right = new TreeNode(4);
	root->right = new TreeNode(5);
	root->right->right = new TreeNode(6);
	flatten(root);
	vector<int> res = inorderTraversal(root);
	printVector(res);
}

TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
	if (preorder.size() == 0)
		return nullptr;
	int rootval = preorder[0];
	TreeNode* root = new TreeNode(rootval);
	int midindex = 0;
	for (int i = 0; i < inorder.size(); i++)
	{
		if (inorder[i] == rootval)
		{
			midindex = i;
			break;
		}
	}
	vector<int> leftinorder(inorder.begin(), inorder.begin() + midindex);
	vector<int> rightinorder(inorder.begin() + midindex + 1, inorder.end());
	vector<int> leftpreorder(preorder.begin()+1, preorder.begin() + 1+leftinorder.size());
	vector<int> rightpreorder(preorder.begin() + 1 + leftinorder.size(), preorder.end());
	root->left = buildTree(leftpreorder, leftinorder);
	root->right = buildTree(rightpreorder, rightinorder);
	return root;
}

void Test_buildTree()
{
	cout << "hot100 105. 从前序与中序遍历序列构造二叉树" << endl;
	vector<int> preorder = {3, 9, 20, 15, 7};
	vector<int> inorder = {9, 3, 15, 20, 7};
	TreeNode* root = buildTree(preorder, inorder);
	vector<vector<int>> res = levelOrder(root);
	printVectorOfVector(res);
}

void find_pathsum(TreeNode* root, int targetSum, long long sum, unordered_map<long long, int>& map, int& res)
{
	if (root == nullptr)
		return;
	sum += root->val;
	res += map.count(sum - targetSum) ? map[sum - targetSum] : 0;
	map[sum]++;
	find_pathsum(root->left, targetSum, sum, map, res);
	find_pathsum(root->right, targetSum, sum, map, res);
	map[sum]--;
}

int pathSum(TreeNode* root, int targetSum) {
	int res = 0;
	unordered_map<long long, int> map;
	map[0] = 1;
	find_pathsum(root, targetSum, 0, map, res);
	return res;
}

void Test_pathSum()
{
	cout << "hot100 437. 路径总和 III" << endl;
	TreeNode* root = new TreeNode(10);
	root->left = new TreeNode(5);
	root->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(-2);
	root->left->right = new TreeNode(2);
	root->left->right->right = new TreeNode(1);
	root->right = new TreeNode(-3);
	root->right->right = new TreeNode(11);
	int res = pathSum(root, 8);
	cout << res << endl;
}

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if (root == nullptr)
		return nullptr;
	if (root == p || root == q)
		return root;
	TreeNode* left = lowestCommonAncestor(root->left, p, q);
	TreeNode* right = lowestCommonAncestor(root->right, p, q);
	if (left && right)
		return root;
	else if (left && !right)
		return left;
	else if (!left && right)
		return right;
	else
		return nullptr;
}

void Test_lowestCommonAncestor()
{
	cout << "hot100 236. 二叉树的最近公共祖先" << endl;
	TreeNode* root = new TreeNode(10);
	root->left = new TreeNode(5);
	root->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(-2);
	root->left->right = new TreeNode(2);
	root->left->right->right = new TreeNode(1);
	root->right = new TreeNode(-3);
	root->right->right = new TreeNode(11);
	TreeNode* res = lowestCommonAncestor(root, root->left->right, root->right->right);
	cout << res->val << endl;
}

int find_maxPathSum(TreeNode* root, int& maxPathSum_sum) {
	if (root == nullptr)
		return 0;
	int leftV = max(0, find_maxPathSum(root->left, maxPathSum_sum));
	int rightV = max(0, find_maxPathSum(root->right, maxPathSum_sum));
	int sumn = root->val + leftV + rightV;
	maxPathSum_sum = max(maxPathSum_sum, sumn);
	return max(leftV, rightV) + root->val;
}

int maxPathSum(TreeNode* root) {
	int maxPathSum_sum = INT_MIN;
	find_maxPathSum(root, maxPathSum_sum);
	return maxPathSum_sum;
}

void Test_maxPathSum()
{
	cout << "hot100 124. 二叉树中的最大路径和" << endl;
	TreeNode* root = new TreeNode(10);
	root->left = new TreeNode(5);
	root->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(3);
	root->left->left->left = new TreeNode(-2);
	root->left->right = new TreeNode(2);
	root->left->right->right = new TreeNode(1);
	root->right = new TreeNode(-3);
	root->right->right = new TreeNode(11);
	int  res = maxPathSum(root);
	cout << res << endl;
}

void numIslands_dfs(vector<vector<char>>& grid, int x, int y, vector<vector<int>> dir)
{
	if (grid[x][y] == '0')
		return;
	if (grid[x][y] == '1')
		grid[x][y] = '0';
	for (int i = 0; i < 4; i++)
	{
		int newx = dir[i][0] + x;
		int newy = dir[i][1] + y;
		if (newx >= 0 && newx < grid.size() && newy >= 0 && newy < grid[0].size())
			numIslands_dfs(grid, newx, newy, dir);
	}
}

int numIslands(vector<vector<char>>& grid) {
	int res = 0;
	vector<vector<int>> dir = { {-1,0},{1,0},{0,1},{0,-1} };
	for (int i = 0; i < grid.size(); i++)
	{
		for (int j = 0; j < grid[0].size(); j++)
		{
			if (grid[i][j] == '1')
			{
				res++;
				numIslands_dfs(grid, i, j, dir);
			}
		}
	}
	return res;
}

void Test_numIslands()
{
	cout << "hot100 200. 岛屿数量" << endl;
	int m, n;
	cout << "\n请输入目标矩阵的高度m和宽度n" << endl;
	cin >> m >> n;
	vector<vector<char>> grid(m, vector<char>(n));
	cout << "\n接下来依次输入矩阵元素:" << endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> grid[i][j];
		}
	}
	int res = numIslands(grid);
	cout << "the res is ：" << res << endl;
}

int orangesRotting(vector<vector<int>>& grid) {
	vector<vector<int>> dir = { {-1,0},{1,0},{0,1},{0,-1} };
	queue<pair<int, int>> que;
	int fresh = 0;
	for (int i = 0; i < grid.size(); i++)
	{
		for (int j = 0; j < grid[0].size(); j++)
		{
			if (grid[i][j] == 1)
				fresh++;
			else if (grid[i][j] == 2)
				que.push({ i,j });
		}
	}
	int res = 0;
	while (!que.empty())
	{
		int size = que.size();
		bool ifbad = false;
		for (int i = 0; i < size; i++)
		{
			int x = que.front().first;
			int y = que.front().second;
			que.pop();
			for (int i = 0; i < dir.size(); i++)
			{
				int newx = x + dir[i][0];
				int newy = y + dir[i][1];
				if (newx >= 0 && newx < grid.size() && newy >= 0 && newy < grid[0].size())
				{
					if (grid[newx][newy] == 1)
					{
						grid[newx][newy] = 2;
						fresh--;
						que.push({ newx,newy });
						ifbad = true;

					}
				}
			}
		}
		if (ifbad)
			res++;
	}
	return fresh == 0 ? res : -1;
}

void Test_orangesRotting()
{
	cout << "hot100 994. 腐烂的橘子" << endl;
	int m, n;
	cout << "\n请输入目标矩阵的高度m和宽度n" << endl;
	cin >> m >> n;
	vector<vector<int>> grid(m, vector<int>(n));
	cout << "\n接下来依次输入矩阵元素(0-空, 1-新鲜, 2-腐烂):" << endl;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> grid[i][j];
		}
	}
	int res = orangesRotting(grid);
	cout << "the res is ：" << res << endl;
}

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	vector<list<int>> graph(numCourses);
	vector<int> input(numCourses);
	for (vector<int> prerequisite : prerequisites)
	{
		graph[prerequisite[1]].push_back(prerequisite[0]);
		input[prerequisite[0]]++;
	}
	queue<int> que;
	for (int i = 0; i<input.size(); i++)
	{
		if (input[i] == 0)
			que.push(i);
	}
	int count = 0;
	while (!que.empty())
	{
		int index = que.front();
		que.pop();
		count++;
		for (int num : prerequisites[index])
		{
			input[num]--;
			if (input[num] == 0)
				que.push(num);
		}
	}
	return count == numCourses;
}

void Test_canFinish()
{
	cout << "hot100 207. 课程表" << endl;
	int numCourses;
	cout << "输入你需要录入多少门课：" << endl;
	cin >> numCourses;
	cout << "依次输入每门课的学习顺序：" << endl;
	vector<vector<int>> prerequisites(numCourses, vector<int>(2));
	for (int i = 0; i < numCourses; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			cin >> prerequisites[i][j];
		}
	}
	bool res = canFinish(numCourses, prerequisites);
	string resstr = (res == true) ? "TRUE" : "FALSE";
	cout << resstr << endl;
}

struct TrieNode
{
	bool isEnd = false;
	TrieNode* next[26];
};


class Trie {
	TrieNode* root;
public:
	Trie() {
		root = new TrieNode();
	}

	void insert(string word) {
		TrieNode* cur = root;
		for (char c : word)
		{
			int idx = c - 'a';
			if (cur->next[idx] == nullptr)
				cur->next[idx] = new TrieNode();
			cur = cur->next[idx];
		}
		cur->isEnd = true;
	}

	bool search(string word) {
		TrieNode* cur = root;
		for (char c : word)
		{
			int idx = c - 'a';
			if (cur->next[idx] == nullptr)
				return false;
			cur = cur->next[idx];
		}
		return cur->isEnd;
	}

	bool startsWith(string prefix) {
		TrieNode* cur = root;
		for (char c : prefix)
		{
			int idx = c - 'a';
			if (cur->next[idx] == nullptr)
				return false;
			cur = cur->next[idx];
		}
		return true;
	}
};

void Test_Trie()
{
	cout << "hot100 208. 实现 Trie (前缀树)" << endl;
	Trie trie;
	trie.insert("apple");
	bool apple = trie.search("apple");
	bool apps = trie.search("app");
	bool appw = trie.startsWith("app");
	trie.insert("app");
	bool appin = trie.search("app");
	cout << apple << " " << apps << " " << appw << " " << appin << endl;
}

void permute_dfs(vector<int>& nums, vector<bool>& visited, vector<vector<int>>& res, vector<int>& temp)
{
	if (temp.size() >= nums.size())
	{
		res.push_back(temp);
		return;
	}
	for (int i = 0; i < nums.size(); i++)
	{
		if (!visited[i])
		{
			visited[i] = true;
			temp.push_back(nums[i]);
			permute_dfs(nums, visited, res, temp);
			temp.pop_back();
			visited[i] = false;
		}
	}
}

vector<vector<int>> permute(vector<int>& nums) {
	vector<bool> visited(nums.size(), false);
	vector<vector<int>> res;
	vector<int> temp;
	permute_dfs(nums, visited, res, temp);
	return res;
}

void Test_permute()
{
	cout << "hot100 46. 全排列" << endl;
	vector<int> nums;
	int num;
	cout << "输入不含重复元素的数组：" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<vector<int>> res = permute(nums);
	printVectorOfVector(res);
}


void subsets_dfs(vector<int>& nums, vector<vector<int>>& res, int start, vector<int>& temp)
{
	res.push_back(temp);
	for (int i = start; i < nums.size(); i++)
	{
		temp.push_back(nums[i]);
		subsets_dfs(nums, res, i+1, temp);
		temp.pop_back();
	}
}

vector<vector<int>> subsets(vector<int>& nums) {
	vector<vector<int>> res;
	vector<int> temp;
	subsets_dfs(nums, res, 0 ,temp);
	return res;
}


void Test_subsets()
{
	cout << "hot100 78. 子集" << endl;
	vector<int> nums;
	int num;
	cout << "输入不含重复元素的数组：" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<vector<int>> res = subsets(nums);
	printVectorOfVector(res);
}


unordered_map<char, string> phoneNum = {
		{'1', ""},    {'2', "abc"},  {'3', "def"}, {'4', "ghi"},  {'5', "jkl"},
		{'6', "mno"}, {'7', "pqrs"}, {'8', "tuv"}, {'9', "wxyz"}, {'0', ""} };

void letterCombinations_dfs(string digits, int star, vector<string>& res,
	string& temp) {
	if (star == digits.size())
		res.push_back(temp);
	string charnum = phoneNum[digits[star]];
	for (char c : charnum) {
		temp.push_back(c);
		letterCombinations_dfs(digits, star + 1, res, temp);
		temp.pop_back();
	}
}

vector<string> letterCombinations(string digits) {
	vector<string> res;
	if (digits.size() == 0) return res;
	string temp;
	letterCombinations_dfs(digits, 0, res, temp);
	return res;
}

void Test_letterCombinations()
{
	cout << "hot100 17. 电话号码的字母组合" << endl;
	string digits;
	cout << "输入数字字符串：" << endl;
	cin >> digits;
	vector<string> res = letterCombinations(digits);
	printVector(res);
}

void combinationSum_dfs(vector<int>& candidates, int target, int idx, vector<vector<int>>& res, vector<int>& temp)
{
	if (target == 0)
		res.push_back(temp);
	for (int i = idx; i < candidates.size(); i++)
	{
		if (target - candidates[i] >= 0)
		{
			temp.push_back(candidates[i]);
			combinationSum_dfs(candidates, target - candidates[i], i, res, temp);
			temp.pop_back();
		}
	}
}

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
	vector<vector<int>> res;
	vector<int> temp;
	combinationSum_dfs(candidates, target, 0, res, temp);
	return res;
}

void Test_combinationSum()
{
	cout << "hot100 39. 组合总和" << endl;
	vector<int> nums;
	int num;
	cout << "\n输入不含重复元素的数组：" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\n输入目标整数target：" << endl;
	cin >> target;
	vector<vector<int>> res = combinationSum(nums, target);
	printVectorOfVector(res);
}

void generateParenthesis_dfs(int left, int right, vector<string>& res, string& temp)
{
	if (left > right)
		return;
	else if (left < 0 || right < 0)
		return;
	else if (left == 0 && right == 0)
		res.push_back(temp);

	temp.push_back('(');
	generateParenthesis_dfs(left - 1, right, res, temp);
	temp.pop_back();

	temp.push_back(')');
	generateParenthesis_dfs(left, right - 1, res, temp);
	temp.pop_back();
}

vector<string> generateParenthesis(int n) {
	vector<string> res;
	string temp;
	generateParenthesis_dfs(n, n, res, temp);
	return res;
}

void Test_generateParenthesis()
{
	cout << "hot100 22. 括号生成" << endl;
	int target;
	cout << "\n输入括号对数：" << endl;
	cin >> target;
	vector<string> res = generateParenthesis(target);
	printVector(res);
}

vector<vector<int>> exist_dir = { {1,0},{-1,0},{0,1},{0,-1} };
bool exist_res = false;
void exist_dfs(vector<vector<char>>& board, string word, int x, int y, int idx, vector<vector<bool>>& visited)
{
	if (board[x][y] != word[idx])
	{
		return;
	}
	else if (board[x][y] == word[idx] && idx == word.size() - 1)
	{
		exist_res = true;
		return;
	}
	for (int i = 0; i < exist_dir.size(); i++)
	{
		int newx = x + exist_dir[i][0];
		int newy = y + exist_dir[i][1];
		if (newx >= 0 && newx < board.size() && newy >= 0 && newy < board[0].size() && !visited[newx][newy])
		{
			visited[newx][newy] = true;
			exist_dfs(board, word, newx, newy, idx + 1, visited);
			visited[newx][newy] = false;
		}
	}
	return;
}

bool exist(vector<vector<char>>& board, string word) {
	vector<vector<bool>> visited(board.size(), vector<bool>(board[0].size(), false));
	for (int i = 0; i < board.size(); i++)
	{
		for (int j = 0; j < board[0].size(); j++)
		{
			if (board[i][j] == word[0])
			{
				visited[i][j] = true;
				exist_dfs(board, word, i, j, 0, visited);
				visited[i][j] = false;
				if (exist_res)
					return true;
			}
		}
	}
	return exist_res;
}


void Test_exist()
{
	cout << "hot100 79. 单词搜索" << endl;
	vector<vector<char>> board = {
		{'A','B','C','E'},
		{'S','F','C','S'},
		{'A','D','E','E'}
	};
	string word = "ABCCED";
	bool res = exist(board, word);
	string resstr = (res == true) ? "true" : "false";
	cout << resstr << endl;
}

bool ifhuiwen(string s, int left, int right)
{
	int begin = left, end = right;
	while (begin < end)
	{
		if (s[begin] != s[end])
			return false;
		begin++;
		end--;
	}
	return true;
}

void partition_dfs(string& s, int idx, vector<vector<string>>& res, vector<string>& temp)
{
	if (idx == s.size())
		res.push_back(temp);
	for (int i = idx; i < s.size(); i++)
	{
		if (ifhuiwen(s, idx, i))
		{
			string stemp = s.substr(idx, i - idx + 1);
			temp.push_back(stemp);
			partition_dfs(s, i + 1, res, temp);
			temp.pop_back();
		}
	}
}

vector<vector<string>> partition(string s) {
	vector<vector<string>> res;
	vector<string> temp;
	partition_dfs(s, 0, res, temp);
	return res;
}

void Test_partition()
{
	cout << "hot100 131. 分割回文串" << endl;
	cout << "\n输入字符串:" << endl;
	string s;
	cin >> s;
	vector<vector<string>> res = partition(s);
	printVectorOfVector(res);
}

bool ifNQueens(int n, int row, int col, vector<string> temp)
{
	for (int i = 0; i < row; i++)
	{
		if (temp[i][col] == 'Q')
			return false;
	}
	for (int i = row-1, j = col-1; i >=0&&j>=0; i--,j--)
	{
		if (temp[i][j] == 'Q')
			return false;
	}
	for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++)
	{
		if (temp[i][j] == 'Q')
			return false;
	}
	return true;
}


void findNQueens(int n, int row, vector<string>& temp, vector<vector<string>>& res)
{
	if (row == n)
	{
		res.push_back(temp);
		return;
	}
	for (int j = 0; j < n; j++)
	{
		if (ifNQueens(n, row, j, temp))
		{
			temp[row][j] = 'Q';
			findNQueens(n, row + 1, temp, res);
			temp[row][j] = '.';
		}
	}
}

vector<vector<string>> solveNQueens(int n) {
	vector<vector<string>> res;
	vector<string> temp(n, string(n, '.'));
	findNQueens(n, 0, temp, res);
	return res;
}

void Test_solveNQueens()
{
	cout << "hot100 51. N 皇后" << endl;
	cout << "\n输入棋盘大小n:" << endl;
	int n;
	cin >> n;
	vector<vector<string>> res = solveNQueens(n);
	printVectorOfVector(res);
}

int searchInsert(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1;
	while (left <= right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid] > target)
		{
			right = mid - 1;
		}
		else if (nums[mid] < target)
		{
			left = mid + 1;
		}
		else {
			return mid;
		}
	}
	return left;
}

void Test_searchInsert()
{
	vector<int> nums;
	int num;
	cout << "hot100 35. 搜索插入位置" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\ninput target value:" << endl;
	cin >> target;
	int res = searchInsert(nums, target);
	cout << "the res is:" << res << endl;
}

bool searchMatrix2(vector<vector<int>>& matrix, int target) {
	int m = matrix.size(), n = matrix[0].size();
	int left = 0, right = m * n - 1;
	while (left <= right)
	{
		int mid = left + (right - left) / 2;
		int newx = mid / n;
		int newy = mid % n;
		if (matrix[newx][newy] > target)
		{
			right = mid - 1;
		}
		else if (matrix[newx][newy] < target)
		{
			left = mid + 1;
		}
		else {
			return true;
		}
	}
	return false;
}

void Test_searchMatrix2()
{
	cout << "\nhot100 74. 搜索二维矩阵" << endl;
	int m, n, target;
	cout << "\n请输入矩阵的行数m和列数n:" << endl;
	cin >> m >> n;
	cout << "\n依次从左上角到右下角输入矩阵元素" << endl;
	vector<vector<int>> nums(m, vector<int>(n, 0));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	cout << "\n请输入目标值target:" << endl;
	cin >> target;

	bool res = searchMatrix2(nums, target);
	string resstr = (res == true) ? "true" : "false";
	cout << resstr << endl;
}

vector<int> searchRange(vector<int>& nums, int target) {
	vector<int> res(2, -1);
	int left = 0, right = nums.size() - 1;
	while (left <= right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid] >= target)
			right = mid - 1;
		else
			left = mid + 1;
	}
	if (left < nums.size() && nums[left] == target)
		res[0] = left;

	left = 0;
	right = nums.size() - 1;
	while (left <= right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid] <= target)
			left = mid + 1;
		else
			right = mid - 1;
	}
	if (right >= 0 && nums[right] == target)
		res[1] = right;

	return res;
}

void Test_searchRange()
{
	vector<int> nums;
	int num;
	cout << "hot100 34. 在排序数组中查找元素的第一个和最后一个位置" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\ninput target value:" << endl;
	cin >> target;
	vector<int> res = searchRange(nums, target);
	printVector(res);
}

int search(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1;
	while (left < right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid] == target)
			return mid;
		if (nums[left] <= nums[mid])
		{
			if (target >= nums[left] && target < nums[mid])
				right = mid - 1;
			else
				left = mid + 1;
		}
		else {
			if (target > nums[mid] && target <= nums[right])
				left = mid + 1;
			else
				right = mid - 1;
		}
	}
	return -1;
}

void Test_search()
{
	vector<int> nums;
	int num;
	cout << "hot100 33. 搜索旋转排序数组" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\ninput target value:" << endl;
	cin >> target;
	int res = search(nums, target);
	cout << res << endl;
}

int findMin(vector<int>& nums) {
	int left = 0, right = nums.size() - 1;
	while (left < right)
	{
		int mid = left + (right - left) / 2;
		if (nums[mid] > nums[right])
			left = mid + 1;
		else
			right = mid;
	}
	return nums[left];
}

void Test_findMin()
{
	vector<int> nums;
	int num;
	cout << "hot100 153. 寻找旋转排序数组中的最小值" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = findMin(nums);
	cout << res << endl;
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
	int m = nums1.size(), n = nums2.size();
	if (m > n)
		return findMedianSortedArrays(nums2, nums1);
	int left = 0, right = m;
	int totalleft = (m + n + 1) / 2; //i+j = m-i + n-j +1
	int leftmax = 0, rightmin = 0;
	while (left <= right)
	{
		int mid_i = left + (right - left) / 2;
		int j = totalleft - mid_i;
		int numi_1 = mid_i - 1 < 0 ? INT_MIN : nums1[mid_i - 1];
		int numj_1 = j - 1 < 0 ? INT_MIN : nums2[j - 1];
		int numi = mid_i >= nums1.size() ? INT_MAX : nums1[mid_i];
		int numj = j >= nums2.size() ? INT_MAX : nums2[j];
		if (numi_1 <= numj)
		{
			leftmax = max(numi_1, numj_1);
			rightmin = min(numi, numj);
			left = mid_i + 1;
		}
		else {
			right = mid_i - 1;
		}
	}
	return (m + n) % 2 == 0 ? (leftmax + rightmin) / 2.0 : leftmax;
}

void Test_findMedianSortedArrays()
{
	vector<int> nums1;
	int num;
	cout << "hot100 4. 寻找两个正序数组的中位数" << endl;
	cout << "\ninput array1:" << endl;
	while (cin >> num)
	{
		nums1.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> nums2;
	cout << "\ninput array2:" << endl;
	while (cin >> num)
	{
		nums2.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	double res = findMedianSortedArrays(nums1, nums2);
	cout << res << endl;
}

bool isValid(string s) {
	stack<char> sta;
	for (char c : s)
	{
		if (c == '(' || c == '[' || c == '{')
		{
			if (c == '(')
				sta.push(')');
			else if (c == '[')
				sta.push(']');
			else if (c == '{')
				sta.push('}');
		}
		else if (c == ')' || c == ']' || c == '}')
		{
			if (sta.empty())
				return false;
			char temp = sta.top();
			sta.pop();
			if (c != temp)
			{
				return false;
			}
		}
	}
	if (!sta.empty())
		return false;
	return true;
}

void Test_isValid()
{
	string s;
	cout << "hot100 20. 有效的括号" << endl;
	cout << "\ninput'('，')'，'{'，'}'，'['，']'" << endl;
	cin >> s;
	bool res = isValid(s);
	string resstr = (res == true) ? "true" : "false";
	cout << resstr << endl;
}

class MinStack {
	stack<int> sta;
	stack<int> minsta;
public:
	MinStack() {

	}

	void push(int val) {
		sta.push(val);
		if (minsta.empty())
			minsta.push(val);
		else
		{
			minsta.push(min(minsta.top(), val));
		}
	}

	void pop() {
		sta.pop();
		minsta.pop();
	}

	int top() {
		return sta.top();
	}

	int getMin() {
		return minsta.top();
	}
};

void Test_MinStack()
{
	MinStack minstack;
	minstack.push(-2);
	minstack.push(0);
	minstack.push(-3);
	int res = minstack.getMin();
	if (res == -3)
		cout << "true" <<endl;
	else
		cout << "false" << endl;
	minstack.pop();
	int a = minstack.top();
	if (a == 0)
		cout << "true" << endl;
	else
		cout << "false" << endl;
	int c = minstack.getMin();
	if (c == -2)
		cout << "true" << endl;
	else
		cout << "false"<< endl;
}

string decodeString(string s) {
	stack<int> nums;
	stack<string> strs;
	int num = 0;
	string res;
	for (char c : s)
	{
		if (c - '0' >= 0 && c - '0' <= 9)
		{
			num = num * 10 + (c - '0');
		}
		else if (c >= 'a' && c <= 'z')
		{
			res += c;
		}
		else if (c == '[')
		{
			nums.push(num);
			num = 0;
			strs.push(res);
			res = "";
		}
		else if (c == ']')
		{
			string tempword = res;
			int times = nums.top();
			nums.pop();
			res = strs.top();
			strs.pop();
			for (int i = 0; i < times; i++)
			{
				res += tempword;
			}
		}
	}
	return res;
}

void Test_decodeString()
{
	string s;
	cout << "hot100 394. 字符串解码" << endl;
	cout << "\ninput string with'['，']' and num:" << endl;
	cin >> s;
	string res = decodeString(s);
	cout << res << endl;
}

vector<int> dailyTemperatures(vector<int>& temperatures) {
	int n = temperatures.size();
	vector<int> res(n);
	stack<int> sta;
	for (int i = 0; i < n; i++)
	{
		while (!sta.empty() && temperatures[i] > temperatures[sta.top()])
		{
			res[sta.top()] = i - sta.top();
			sta.pop();
		}
		sta.push(i);
	}
	return res;
}

void Test_dailyTemperatures()
{
	vector<int> nums;
	int num;
	cout << "hot100 739. 每日温度" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> res = dailyTemperatures(nums);
	printVector(res);
}

int largestRectangleArea(vector<int>& heights) {
	/*	暴力解法
		int res = 0;
        for(int i = 0 ; i<heights.size(); i++)
        {
            int leftidx = i;
            int rightidx = i;
            while(leftidx>=0&&heights[leftidx]>=heights[i])
            {
                leftidx--;
            }
            while(rightidx<heights.size()&&heights[rightidx]>=heights[i])
            {
                rightidx++;
            }
            res = max(res, heights[i]*(rightidx-leftidx-1));
        }
        return res;
	*/
	int n = heights.size();
	vector<int> left(n), right(n);
	stack<int> sta;
	for (int i = 0; i < n; i++) {
		while (!sta.empty() && heights[i] <= heights[sta.top()]) {
			sta.pop();
		}
		left[i] = sta.empty() ? -1 : sta.top();
		sta.push(i);
	}
	sta = stack<int>();
	for (int i = n - 1; i >= 0; i--) {
		while (!sta.empty() && heights[i] <= heights[sta.top()]) {
			sta.pop();
		}
		right[i] = sta.empty() ? n : sta.top();
		sta.push(i);
	}
	int res = 0;
	for (int i = 0; i < n; i++) {
		res = max(res, heights[i] * (right[i] - left[i] - 1));
	}
	return res;
}

void Test_largestRectangleArea()
{
	vector<int> nums;
	int num;
	cout << "hot100 84. 柱状图中最大的矩形" << endl;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = largestRectangleArea(nums);
	cout << "the res is:" << res << endl;
}


int findKthLargest(vector<int>& nums, int k) {
	priority_queue<int, vector<int>> que;
	for (int i : nums) {
		que.push(i);
		/*
		* if(que.size()>k)
		*	que.pop();
		*/
	}
	for (int i = 0; i < k - 1; i++) {
		que.pop();
	}
	return que.top();
}

void Test_findKthLargest()
{
	int target;
	cout << "hot100 215. 数组中的第K个最大元素" << endl;
	cout << "\ninput the k:" << endl;
	cin >> target;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = findKthLargest(nums, target);
	cout << "\nthe result:"<<res<< endl;
}

vector<int> topKFrequent(vector<int>& nums, int k) {
	auto duibi = [](pair<int, int>& a, pair<int, int>& b) {
		return a.second > b.second;
	};
	priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(duibi)>
		que(duibi);
	unordered_map<int, int> times;
	for (int num : nums)
		times[num]++;
	for (unordered_map<int, int>::iterator it = times.begin();
		it != times.end(); it++) {
		que.push(*it);
		if (que.size() > k) {
			que.pop();
		}
	}
	vector<int> res;
	while (k--) {
		res.push_back(que.top().first);
		que.pop();
	}
	return res;
}

void Test_topKFrequent()
{
	int target;
	cout << "hot100 347. 前 K 个高频元素" << endl;
	cout << "\ninput the k:" << endl;
	cin >> target;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> res = topKFrequent(nums, target);
	printVector(res);
}

class MedianFinder {
private:
	priority_queue<int, vector<int>> que1;
	priority_queue<int, vector<int>, greater<int>> que2;

public:
	MedianFinder() {

	}

	void addNum(int num) {
		if (que1.size() > que2.size())
		{
			que1.push(num);
			double n = que1.top();
			que1.pop();
			que2.push(n);
		}
		else {
			que2.push(num);
			double m = que2.top();
			que2.pop();
			que1.push(m);
		}
	}

	double findMedian() {
		if (que1.size() > que2.size())
		{
			return double(que1.top());
		}
		else if (que1.size() < que2.size()) {
			return double(que2.top());
		}
		else {
			return (que1.top() + que2.top()) / 2.0;
		}
	}
};

void Test_MedianFinder()
{
	cout << "hot100 295. 数据流的中位数" << endl;
	MedianFinder medianfinder;
	medianfinder.addNum(1);
	medianfinder.addNum(2);
	cout << medianfinder.findMedian() << endl;
	medianfinder.addNum(3);
	cout << medianfinder.findMedian() << endl;
}


int maxProfit(vector<int>& prices) {
	int res = 0;
	int pre = prices[0];
	for (int i = 1; i < prices.size(); i++)
	{
		pre = min(pre, prices[i]);
		res = max(res, (prices[i] - pre));
	}
	return res;
}

void Test_maxProfit()
{
	cout << "hot121. 买卖股票的最佳时机" << endl;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = maxProfit(nums);
	cout << res << endl;
}

bool canJump(vector<int>& nums) {
	int pace = 0;
	for (int i = 0; i < nums.size() - 1; i++) {
		pace = max(pace, nums[i] + i);
		if (pace <= i)
			return false;
	}
	return pace >= nums.size() - 1;
}

void Test_canJump()
{
	cout << "hot55. 跳跃游戏" << endl;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	bool res = canJump(nums);
	string strres = (res == true) ? "true" : "false";
	cout << strres << endl;
}

int jump(vector<int>& nums) {
	int idx = 0;
	int maxfar = 0;
	int end = 0;
	for (int i = 0; i < nums.size() - 1; i++)
	{
		maxfar = max(maxfar, nums[i] + i);
		if (i == end)
		{
			idx++;
			end = maxfar;
		}
	}
	return idx;
}

void Test_jump()
{
	cout << "hot45. 跳跃游戏 II" << endl;
	vector<int> nums;
	int num;
	cout << "input array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = jump(nums);
	cout << "The res is :" << res << endl;
}

vector<int> partitionLabels(string s) {
	unordered_map<char, int> map;
	for (int i = 0; i < s.size(); i++)
	{
		map[s[i]] = i;
	}
	vector<int> res;
	int right = 0;
	int left = 0;
	for (int i = 0; i < s.size(); i++)
	{
		right = max(right, map[s[i]]);
		if (i == right)
		{
			res.push_back(right - left + 1);
			left = right + 1;
		}
	}
	return res;
}

void Test_partitionLabels()
{
	cout << "763. 划分字母区间" << endl;
	string inputstr;
	cout << "input array:" << endl;
	cin >> inputstr;
	vector<int> res = partitionLabels(inputstr);
	cout << "\nThe res is :" << endl;
	printVector(res);
}

int climbStairs(int n) {
	vector<int> dp(n + 1, 0);
	dp[0] = 1;
	dp[1] = 1;
	for (int i = 2; i <= n; i++)
	{
		dp[i] = dp[i - 1] + dp[i - 2];
	}
	return dp[n];
}

void Test_climbStairs()
{
	cout << "70. 爬楼梯" << endl;
	int n;
	cout << "\ninput n:" << endl;
	cin >> n;
	int res = climbStairs(n);
	cout << "\nThe res is :" << res << endl;
}

vector<vector<int>> generate(int numRows) {
	vector<vector<int>> res;
	for (int i = 0; i < numRows; i++)
	{
		vector<int> temp(i + 1, 1);
		res.push_back(temp);
	}
	for (int i = 2; i < numRows; i++)
	{
		for (int j = 1; j < i; j++)
		{
			res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
		}
	}
	return res;
}

void Test_generate()
{
	cout << "118. 杨辉三角" << endl;
	int n;
	cout << "\ninput n:" << endl;
	cin >> n;
	vector<vector<int>> res = generate(n);
	cout << "\nThe res is :" << endl;
	printVectorOfVector(res);
}

int rob(vector<int>& nums) {
	int n = nums.size();
	vector<int> dp(n + 1, 0);
	dp[0] = 0;  dp[1] = nums[0];
	for (int i = 2; i <= n; i++)
	{
		dp[i] = max(nums[i - 1] + dp[i - 2], dp[i - 1]);//偷，不偷
	}
	return dp[n];
}

void Test_rob()
{
	cout << "198. 打家劫舍" << endl;
	vector<int> nums;
	int num;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = rob(nums);
	cout << "\nThe res is :" << res << endl;
}

int numSquares(int n) {
	vector<int> dp(n + 1, INT_MAX);
	dp[0] = 0;
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j * j <= i; j++) {
			dp[i] = min(dp[i], dp[i - j * j] + 1);
		}
	}
	return dp[n];
}

void Test_numSquares()
{
	cout << "279. 完全平方数" << endl;
	int n;
	cout << "\ninput n:" << endl;
	cin >> n;
	int res = numSquares(n);
	cout << "\nThe res is :" << res <<endl;
}

int coinChange(vector<int>& coins, int amount) {
	vector<int> dp(amount + 1, amount + 1);
	dp[0] = 0;
	for (int i = 1; i <= amount; i++)
	{
		for (int j = 0; j < coins.size(); j++)
		{
			if (i - coins[j] < 0)
				continue;
			//if(dp[i-coins[j]]!=INT_MAX)
			dp[i] = min(dp[i], dp[i - coins[j]] + 1);
		}
	}
	if (dp[amount] == amount + 1)
		return -1;
	return dp[amount];
}

void Test_coinChange()
{
	cout << "322. 零钱兑换" << endl;
	vector<int> nums;
	int num;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int amount;
	cout << "\ninput amount:" << endl;
	cin >> amount;
	int res = coinChange(nums, amount);
	cout << "\nThe res is :" << res << endl;
}

bool wordBreak(string s, vector<string>& wordDict) {
	unordered_set<string> myset(wordDict.begin(), wordDict.end());
	vector<bool> dp(s.size()+1, false);
	dp[0] = true;
	for (int i = 1; i <= s.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			string word = s.substr(j, i - j);
			if (myset.find(word) != myset.end() && dp[j])
				dp[i] = true;
		}
	}
	return dp[s.size()];
}

void Test_wordBreak()
{
	cout << "139. 单词拆分" << endl;
	string s;
	cout << "\ninput s:" << endl;
	cin >> s;
	vector<string> nums;
	string temp;
	cout << "\ninput wordDict:" << endl;
	while (cin >> temp)
	{
		nums.push_back(temp);
		if (cin.peek() == '\n')
			break;
	}
	bool res = wordBreak(s, nums);
	string strres = (res == true) ? "TRUE" : "FALSE";
	cout << "\nThe res is :" << strres << endl;
}

int lengthOfLIS(vector<int>& nums) {
	vector<int> dp(nums.size(), 1);
	for (int i = 1; i < nums.size(); i++)
	{
		for (int j = 0; j < i; j++)
		{
			if (nums[j] < nums[i])
				dp[i] = max(dp[i], dp[j] + 1);
		}
	}
	int res = 0;
	for (int num : dp)
		res = max(num, res);
	return res;
}

void Test_lengthOfLIS()
{
	cout << "300. 最长递增子序列" << endl;
	vector<int> nums;
	int num;
	cout << "\ninput array:" << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = lengthOfLIS(nums);
	cout << "\nThe res is :" << res << endl;
}

int maxProduct(vector<int>& nums) {
	int n = nums.size();
	vector<int> maxv(n), minv(n);
	maxv[0] = nums[0];
	minv[0] = nums[0];
	for (int i = 1; i < n; i++)
	{
		int num = nums[i];
		maxv[i] = max({ maxv[i - 1] * num, minv[i - 1] * num, num });
		minv[i] = min({ maxv[i - 1] * num, minv[i - 1] * num, num });
	}
	int res = INT_MIN;
	for (int num : maxv)
		res = max(res, num);
	return res;
}

void Test_maxProduct()
{
	cout << "152. 乘积最大子数组" << endl;
	vector<int> nums;
	int num;
	cout << "\n请输入数组: " << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = maxProduct(nums);
	cout << "\nThe res is : " << res << endl;
}

bool canPartition(vector<int>& nums) {
	int sum = 0;
	for (int num : nums)
		sum += num;
	if (sum % 2 != 0)
		return false;
	int target = sum / 2;
	int n = nums.size();
	vector<vector<bool>> dp(n + 1, vector<bool>(target + 1, false));
	for (int i = 0; i <= n; i++)
		dp[i][0] = true;
	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= target; j++)
		{
			if (j - nums[i-1] < 0)
				dp[i][j] = dp[i - 1][j];
			else
				dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i-1]];
		}
	}
	return dp[n][target];
}

void Test_canPartition()
{
	cout << "416. 分割等和子集" << endl;
	vector<int> nums;
	int num;
	cout << "\n请输入数组: " << endl;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	bool res = canPartition(nums);
	string str = (res == true) ? "true" : "false";
	cout << "\nThe res is : " << str << endl;
}

int longestValidParentheses(string s) {
	vector<int> dp(s.size(), 0);
	stack<int> sta;
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == '(')
			sta.push(i);
		else {
			if (!sta.empty())
			{
				int leftidx = sta.top();
				sta.pop();
				int length = i - leftidx + 1;
				if (leftidx - 1 >= 0)
					length += dp[leftidx - 1];
				dp[i] = length;
			}
		}
	}
	int res = 0;
	for (int num : dp)
		res = max(res, num);
	return res;
}

void Test_longestValidParentheses()
{
	cout << "32. 最长有效括号" << endl;
	string s;
	cout << "\ninput the string by using '(', ')' " << endl;
	cin >> s;
	int res = longestValidParentheses(s);
	cout << "\nThe res is : " << res << endl;
}

int main()
{
	//Test_twosum();						/*			hot100 1.	两数之和									*/
	//Test_groupAnagrams();					/*			hot100 49.	字母异位词分组							*/
	//Test_longestConsecutive();			/*			hot100 128.	最长连续序列								*/
	//Test_moveZeroes();					/*			hot100 283.	移动零									*/
	//Test_maxArea();						/*			hot100 11.	盛最多水的容器							*/
	//Test_threeSum();						/*			hot100 15.	三数之和									*/
	//Test_trap();							/*			hot100 42.	接雨水									*/
	//Test_lengthOfLongestSubstring();		/*			hot100 3.	无重复字符的最长子串						*/
	//Test_findAnagrams();					/*			hot100 438.	找到字符串中所有字母异位词					*/
	//Test_subarraySum();					/*			hot100 560. 和为 K 的子数组							*/
	//Test_maxSlidingWindow();				/*			hot100 239. 滑动窗口最大值							*/
	//Test_minWindow();						/*			hot100 76.	最小覆盖子串								*/
	//Test_maxSubArray();					/*			hot100 53.	最大子数组和								*/
	//Test_merge();							/*			hot100 56.	合并区间									*/	
	//Test_rotate();						/*			hot100 189. 轮转数组									*/
	//Test_productExceptSelf();				/*			hot100 238. 除自身以外数组的乘积						*/
	//Test_firstMissingPositive();			/*			hot100 41.	缺失的第一个正数							*/
	//Test_setZeroes();						/*			hot100 73.	矩阵置零									*/
	//Test_spiralOrder();					/*			hot100 54.	螺旋矩阵									*/
	//Test_rotate_nums();					/*			hot100 48.	旋转图像									*/
	//Test_searchMatrix();					/*			hot100 240. 搜索二维矩阵 II							*/
	//Test_getIntersectionNode();			/*			hot100 160. 相交链表									*/
	//Test_reverseList();					/*			hot100 206. 反转链表									*/
	//Test_isPalindrome();					/*			hot100 234. 回文链表									*/
	//Test_hasCycle();						/*			hot100 141. 环形链表									*/
	//Test_detectCycle();					/*			hot100 142. 环形链表 II								*/
	//Test_mergeTwoLists();					/*			hot100 21.	合并两个有序链表							*/
	//Test_addTwoNumbers();					/*			hot100 2.	两数相加									*/
	//Test_removeNthFromEnd();				/*			hot100 19.	删除链表的倒数第 N 个结点					*/
	//Test_swapPairs();						/*			hot100 24.	两两交换链表中的节点						*/
	//Test_reverseKGroup();					/*			hot100 25.	K 个一组翻转链表							*/
	//Test_copyRandomList();				/*			hot100 138. 随机链表的复制							*/
	//Test_sortList();						/*			hot100 148. 排序链表									*/
	//Test_mergeKLists();					/*			hot100 23.	合并 K 个升序链表							*/
	//Test_LRUCache();						/*			hot100 146. LRU 缓存									*/
	//Test_inorderTraversal();				/*			hot100 94.  二叉树的中序遍历							*/
	//Test_maxDepth();						/*			hot100 104. 二叉树的最大深度							*/
	//Test_invertTree();					/*			hot100 226. 翻转二叉树								*/
	//Test_isSymmetric();					/*			hot100 101. 对称二叉树								*/
	//Test_diameterOfBinaryTree();			/*			hot100 543. 二叉树的直径								*/
	//Test_levelOrder();					/*			hot100 102. 二叉树的层序遍历							*/
	//Test_sortedArrayToBST();				/*			hot100 108. 将有序数组转换为二叉搜索树					*/
	//Test_isValidBST();					/*			hot100 98.	验证二叉搜索树							*/
	//Test_kthSmallest();					/*			hot100 230. 二叉搜索树中第 K 小的元素					*/
	//Test_rightSideView();					/*			hot100 199. 二叉树的右视图							*/
	//Test_flatten();						/*			hot100 114. 二叉树展开为链表							*/
	//Test_buildTree();						/*			hot100 105. 从前序与中序遍历序列构造二叉树				*/
	//Test_pathSum();						/*			hot100 437. 路径总和 III								*/
	//Test_lowestCommonAncestor();			/*			hot100 236. 二叉树的最近公共祖先						*/
	//Test_maxPathSum();					/*			hot100 124. 二叉树中的最大路径和						*/
	//Test_numIslands();					/*			hot100 200. 岛屿数量									*/
	//Test_orangesRotting();				/*			hot100 994. 腐烂的橘子								*/
	//Test_canFinish();						/*			hot100 207. 课程表									*/
	//Test_Trie();							/*			hot100 208. 实现 Trie (前缀树)						*/
	//Test_permute();						/*			hot100 46.	全排列									*/
	//Test_subsets();						/*			hot100 78.	子集										*/
	//Test_letterCombinations();			/*			hot100 17.	电话号码的字母组合						*/
	//Test_combinationSum();				/*			hot100 39.	组合总和									*/
	//Test_generateParenthesis();			/*			hot100 22.	括号生成									*/
	//Test_exist();							/*			hot100 79.	单词搜索									*/
	//Test_partition();						/*			hot100 131. 分割回文串								*/
	//Test_solveNQueens();					/*			hot100 51.	N 皇后									*/
	//Test_searchInsert();					/*			hot100 35.  搜索插入位置								*/
	//Test_searchMatrix2();					/*			hot100 74.	搜索二维矩阵								*/
	//Test_searchRange();					/*			hot100 34.	在排序数组中查找元素的第一个和最后一个位置  */
	//Test_search();						/*			hot100 33.	搜索旋转排序数组							*/
	//Test_findMin();						/*			hot100 153. 寻找旋转排序数组中的最小值					*/
	//Test_findMedianSortedArrays();		/*			hot100 4.	寻找两个正序数组的中位数					*/
	//Test_isValid();						/*			hot100 20.	有效的括号								*/
	//Test_MinStack();						/*			hot100 155. 最小栈									*/
	//Test_decodeString();					/*			hot100 394. 字符串解码								*/
	//Test_dailyTemperatures();				/*			hot100 739. 每日温度									*/
	//Test_largestRectangleArea();			/*			hot100 84.	柱状图中最大的矩形						*/
	//Test_findKthLargest();				/*			hot100 215. 数组中的第K个最大元素						*/
	//Test_topKFrequent();					/*			hot100 347. 前 K 个高频元素							*/
	//Test_MedianFinder();					/*			hot100 295. 数据流的中位数							*/
	//Test_maxProfit();						/*			hot100 121. 买卖股票的最佳时机						*/
	//Test_canJump();						/*			hot100 55.	跳跃游戏									*/
	//Test_jump();							/*			hot100 45.	跳跃游戏 II								*/
	//Test_partitionLabels();				/*			hot100 763. 划分字母区间								*/
	//Test_climbStairs();					/*			hot100 70.	爬楼梯									*/
	//Test_generate();						/*			hot100 118. 杨辉三角									*/
	//Test_rob();							/*			hot100 198. 打家劫舍									*/
	//Test_numSquares();					/*			hot100 279. 完全平方数								*/
	//Test_coinChange();					/*			hot100 322. 零钱兑换									*/
	//Test_wordBreak();						/*			hot100 139. 单词拆分									*/
	//Test_lengthOfLIS();					/*			hot100 300. 最长递增子序列							*/
	//Test_maxProduct();					/*			hot100 152. 乘积最大子数组							*/
	//Test_canPartition();					/*			hot100 416. 分割等和子集								*/
	//Test_longestValidParentheses();		/*			hot100	32. 最长有效括号								*/
}	