#include<iostream>
#include<vector>
#include <unordered_map>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <unordered_set>
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
	cout << "hot100 1.����֮��" << endl;
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
	cout << "hot100 49.��ĸ��λ�ʷ���" << endl;
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
	for (int num:myset)//set�Զ�ȥ�أ�����nums��������ظ���Ԫ��
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
	cout << "hot100 128.���������" << endl;
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
	cout << "hot100 283.�ƶ���" << endl;
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
	cout << "hot100 11.ʢ���ˮ������" << endl;
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
				//��������ǰһ����ȣ��ͼ�����ǰ��, ����һ��Ҫleft<right
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
	cout << " hot100 15.����֮�� " << endl;
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
	cout << " hot100 42.����ˮ " << endl;
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
	cout << " hot100 3.���ظ��ַ�����Ӵ� " << endl;
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
	cout << "438. �ҵ��ַ�����������ĸ��λ��" << endl;
	cout << "\n���볤�ַ�����" << endl;
	cin >> s;
	cout << "\n������λ�ʣ�" << endl;
	cin >> p;
	vector<int> res = findAnagrams(s, p);
	cout << "\n����Ľ���ǣ�" << endl;
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
	cout << "560. ��Ϊ K ��������" << endl;
	cout << "\n��������nums��" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\n����target��" << endl;
	cin >> target;
	
	int res = subarraySum(nums, target);
	cout << "\n����Ľ���ǣ�"<< res << endl;
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
	cout << "239. �����������ֵ" << endl;
	cout << "\n��������nums��" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int target;
	cout << "\n����target��" << endl;
	cin >> target;

	vector<int> res = maxSlidingWindow(nums, target);
	cout << "\n����Ľ���ǣ�" <<endl;
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
	cout << "76. ��С�����Ӵ�" << endl;
	cout << "\n���볤�ַ�����" << endl;
	cin >> s;
	cout << "\n������ַ�����" << endl;
	cin >> p;
	string res = minWindow(s, p);
	cout << "\n����Ľ���ǣ�" << endl;
	cout << res << endl;
}

int maxSubArray(vector<int>& nums) {
#if 1
	cout << "\n��̬�滮��" << endl;
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
	cout << "\n̰�ģ�" << endl;
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
	cout << "53. ����������" << endl;
	cout << "\n��������nums��" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = maxSubArray(nums);
	cout << "\n����Ľ���ǣ�" <<res<< endl;
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
	cout << "56. �ϲ�����" << endl;
	cout << "\n�����������䣨a, b����" << endl;
	cout << "����������ַ���������" << endl;
	vector<vector<int>> nums;
	int a, b;
	while (cin >> a >> b)
	{
		nums.push_back({a, b});
	}
	vector<vector<int>> res = merge(nums);
	cout << "\n����Ľ���ǣ�" << endl;
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
	cout << "189. ��ת����" << endl;
	cout << "\n������������Ԫ�أ�" << endl;
	vector<int> nums;
	int num;
	int k;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	cout << "\n����������ת��λ��k��" << endl;
	cin >> k;
	rotate(nums, k);
	cout << "\n����Ľ���ǣ�" << endl;
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
	cout << "238. ��������������ĳ˻�" << endl;
	cout << "\n������������Ԫ�أ�" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	vector<int> res = productExceptSelf(nums);
	cout << "\n����Ľ���ǣ�" << endl;
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
	cout << "41. ȱʧ�ĵ�һ������" << endl;
	cout << "\n������������Ԫ�أ�" << endl;
	vector<int> nums;
	int num;
	while (cin >> num)
	{
		nums.push_back(num);
		if (cin.peek() == '\n')
			break;
	}
	int res = firstMissingPositive(nums);
	cout << "\n����Ľ���ǣ�" << res << endl;
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
	cout << "73. ��������" << endl;
	int m, n;
	cout << "\n������������m������n��" << endl;
	cin >> m >> n;
	cout << "�����ϵ����������������Ԫ��" << endl;
	vector<vector<int>> nums(m, vector<int>(n));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	setZeroes(nums);
	cout << "\n����Ľ���ǣ�" << endl;
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
	cout << "\n��������������m������n:" << endl;
	cin >> m >> n;
	cout << "\n���δ����Ͻǵ����½��������Ԫ��" << endl;
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
	cout << "\n������n*n����Ĵ�Сn:" << endl;
	cin >> n;
	cout << "\n���δ����Ͻǵ����½��������Ԫ��" << endl;
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
	cout << "\n��������������m������n:" << endl;
	cin >> m >> n;
	cout << "\n���δ����Ͻǵ����½��������Ԫ��" << endl;
	vector<vector<int>> nums(m, vector<int>(n, 0));
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> nums[i][j];
		}
	}
	cout << "\n������Ŀ��ֵtarget:" << endl;
	cin >> target;

	bool res = searchMatrix(nums, target);
	cout << res << endl;
}

int main()
{
	//Test_twosum();						/*			hot100 1.	����֮��						*/
	//Test_groupAnagrams();					/*			hot100 49.	��ĸ��λ�ʷ���				*/
	//Test_longestConsecutive();			/*			hot100 128.	���������					*/
	//Test_moveZeroes();					/*			hot100 283.	�ƶ���						*/
	//Test_maxArea();						/*			hot100 11.	ʢ���ˮ������				*/
	//Test_threeSum();						/*			hot100 15.	����֮��						*/
	//Test_trap();							/*			hot100 42.	����ˮ						*/
	//Test_lengthOfLongestSubstring();		/*			hot100 3.	���ظ��ַ�����Ӵ�			*/
	//Test_findAnagrams();					/*			hot100 438.	�ҵ��ַ�����������ĸ��λ��		*/
	//Test_subarraySum();					/*			hot100 560. ��Ϊ K ��������				*/
	//Test_maxSlidingWindow();				/*			hot100 239. �����������ֵ				*/
	//Test_minWindow();						/*			hot100 76.	��С�����Ӵ�					*/
	//Test_maxSubArray();					/*			hot100 53.	����������					*/
	//Test_merge();							/*			hot100 56.	�ϲ�����						*/	
	//Test_rotate();						/*			hot100 189. ��ת����						*/
	//Test_productExceptSelf();				/*			hot100 238. ��������������ĳ˻�			*/
	//Test_firstMissingPositive();			/*			hot100 41.	ȱʧ�ĵ�һ������				*/
	//Test_setZeroes();						/*			hot100 73.	��������						*/
	//Test_spiralOrder();					/*			hot100 54.	��������						*/
	//Test_rotate_nums();					/*			hot100 48.	��תͼ��						*/
	//Test_searchMatrix();					/*			hot100 48.	240. ������ά���� II			*/
}