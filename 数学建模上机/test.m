%% =========== MATLAB intlinprog 示例 =============
clear; clc;

% -------- 1. 设置基本参数 --------
num_days = 7;       % 一周7天
num_shifts = 5;     % 5个班次

% 每个时间段的需求(这里是一个简化示例；题目中要细分每个时段)
demand_6_10  = 18;
demand_10_14 = 20;
demand_14_18 = 19;
demand_18_22 = 17;
demand_22_6  = 12;

% 若用思路A，则决策变量 x(d,s) 的个数为 7*5 = 35
% x(1,1) ~ x(7,5). 这里线性索引后长度为 nVars = 35
nVars = num_days * num_shifts;

% -------- 2. 构造目标函数 --------
% 目标: minimize sum(x_{d,s})
f = ones(nVars, 1);

% -------- 3. 构造不等式约束 A*x <= b --------
% (a) 时间段覆盖约束 (这里仅示范 "6:00~10:00" 这一条)
%     "6:00~10:00" 由班次1(2:00-10:00)和班次2(6:00-14:00)覆盖
%      => x(d,1) + x(d,2) >= 18 for each day d=1..7
%  在 intlinprog 中需要写成 A*x >= b 的形式, 通常转换成 -A*x <= -b
A = [];
b = [];

% 举例: 针对 d=1 (周一):
%     x(1,1) + x(1,2) >= 18  =>  -x(1,1) - x(1,2) <= -18
% 我们需要在 A 的某一行, 对应 x(1,1), x(1,2) 位置放 -1, 其余放 0.
% 下面演示一个循环来添加各天 "6:00~10:00" 约束

for d = 1:num_days
    row = zeros(1,nVars);
    % 班次1,2 在 day=d 中的变量索引
    idx1 = (d-1)*num_shifts + 1; % x(d,1) 的线性索引
    idx2 = (d-1)*num_shifts + 2; % x(d,2)
    row(idx1) = -1;
    row(idx2) = -1;
    A = [A; row];    % 添加一行
    b = [b; -demand_6_10];
end

% (b) 其它时间段(10:00~14:00, 14:00~18:00, 18:00~22:00, 22:00~6:00)
%     也需同理循环. 注意 "22:00~6:00" 要跨到下一天 => x(d,5) + x(d+1,1) >= 12
%     要做适当的模运算(d+1 => (d mod 7)+1).

% (c) 每人每周只能上 5 天班, 需要限制 sum_{d} sum_{s} x(d,s) <= 5 * (总护士数)
%     但这里我们要最小化 "总护士数"? 可以在思路A中再加一个变量 y 表示
%     "总护士数", 并添加 x(d,s) 的和 <= y 以及 y 整数最小化. 
%     这一部分略去, 留待论文中结合具体思路实现.

% -------- 4. 指定变量类型(整数) --------
intcon = 1:nVars;   % 所有变量都要求是整数

% -------- 5. 调用 intlinprog 求解 --------
lb = zeros(nVars,1);
ub = inf(nVars,1);
opts = optimoptions('intlinprog','Display','iter');

[x_sol, fval, exitflag] = intlinprog(f,intcon,A,b,[],[],lb,ub,[],opts);

% 之后 x_sol(i) 得到每个 x(d,s) 的最优解, fval 是目标函数值(最小化的总护士数量).
% 根据 exitflag 可以查看是否找到可行解.
