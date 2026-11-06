"""
空间感知 —— 机器人在房间里的真实位置与周围物体

这个模块存在的理由：
在此之前，Animat 能感觉到"我有点无聊"，却不知道能去哪儿。它反复
说"想找点有趣的事做"却始终不动，因为上下文里除了电量体温之外
没有任何关于外部空间的信息——它连自己在哪都不知道，自然给不出
一个目的地。

坐标系与 webui/index.html 的 3D 视图共用同一套：
  x,z 为水平面，y 向上；单位是米；房间半径 BOUND 米。
3D 场景里的物体位置从本模块广播的状态绘制，模型看到的距离由本模块
按真实坐标计算，所以"它说走向书架"和"画面上真的走向书架"是同一件事，
不是两套平行的叙事。

刻意不做的事：
  - 不做路径规划。房间是开阔的，直线走过去就行。
  - 不做障碍物碰撞。现在没有需要绕开的体积物。
  - 不做视觉渲染。这里只是数据，画是前端的事。
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Object:
    """房间里的一件东西。"""
    name: str
    x: float
    z: float
    desc: str
    # 走到多近算"到了"。物体的体积不同，书架要贴近才算到，
    # 窗户隔着几米看一眼也算到了。
    reach: float = 1.2


@dataclass
class World:
    """房间的静态布局。

    位置是写死的常量而非随机生成：一个房间里的家具不会自己长腿跑，
    每次启动位置都一样，模型对它的记忆才有意义。
    """

    radius: float = 17.0        # 可活动半径，与 3D 视图的 BOUND 一致

    objects: list[Object] = field(default_factory=lambda: [
        Object("充电座", 8.0, -6.5, "墙角的充电位，底座上有一圈淡蓝色指示灯", reach=1.6),
        Object("窗户", -10.5, 4.0, "一扇朝北的窗，能看见外面的夜色和远处的灯", reach=2.0),
        Object("书架", 11.5, 8.5, "靠墙的木书架，摆着十几本厚薄不一的书，封面有些旧", reach=1.5),
        Object("沙发", -4.0, -8.0, "一张深灰色的布艺沙发，坐垫有点塌陷", reach=1.4),
        Object("工作台", 3.5, 11.0, "一张工作台，上面散着螺丝刀和几块电路板", reach=1.3),
    ])


class SpatialState:
    """机器人在房间里的当前位置与朝向。"""

    def __init__(self, world: Optional[World] = None):
        self.world = world or World()
        # 初始位置：房间中央偏一点，不在任何物体旁边
        self.x: float = 0.0
        self.z: float = 0.0
        # 朝向角。0 表示朝向 -z（画面里"前方"），顺时针为正。
        self.heading: float = 0.0

        # 当前正在前往的目标，None 表示没在移动。
        # 用于"到地方了"的判定和移动途中的状态描述。
        self.target: Optional[str] = None

    # ── 坐标工具 ──
    def distance_to(self, obj: Object) -> float:
        return math.hypot(obj.x - self.x, obj.z - self.z)

    def bearing_to(self, obj: Object) -> float:
        """物体相对朝向的角度（度）。0=正前，90=右侧，-90=左侧，180=背后。"""
        dx, dz = obj.x - self.x, obj.z - self.z
        # atan2 给出相对 +x 轴的角度，转成本系统的"朝向 -z 为 0"
        absolute = math.degrees(math.atan2(dx, -dz))
        rel = (absolute - self.heading + 180) % 360 - 180
        return rel

    @staticmethod
    def direction_word(rel_deg: float) -> str:
        """把相对角度翻成人话。

        分档而不给精确角度：模型不需要知道"偏右 37 度"，
        知道"右前方"就足够决定往哪走。
        """
        a = abs(rel_deg)
        if a <= 25:
            return "正前方"
        if a <= 65:
            return "右前方" if rel_deg > 0 else "左前方"
        if a <= 115:
            return "右侧" if rel_deg > 0 else "左侧"
        if a <= 155:
            return "右后方" if rel_deg > 0 else "左后方"
        return "正后方"

    def facing_word(self) -> str:
        """当前朝向的方位词，用于让模型知道自己面对着什么。"""
        h = self.heading % 360
        for deg, name in ((0, "北"), (45, "东北"), (90, "东"), (135, "东南"),
                          (180, "南"), (225, "西南"), (270, "西"), (315, "西北")):
            if abs(((h - deg + 180) % 360) - 180) <= 22.5:
                return name
        return "北"

    # ── 感知 ──
    def nearby(self, max_dist: float = 12.0) -> list[tuple[Object, float, float]]:
        """视野内的物体，按距离排序。

        返回 (物体, 距离, 相对角度)。超过 max_dist 的不返回——
        太远的东西不该出现在"你现在能看到什么"里，
        否则模型会以为自己站在一个什么都够得着的地方。
        """
        found = []
        for obj in self.world.objects:
            d = self.distance_to(obj)
            if d <= max_dist:
                found.append((obj, d, self.bearing_to(obj)))
        found.sort(key=lambda t: t[1])
        return found

    def at_object(self) -> Optional[Object]:
        """当前所在的物体旁（在 reach 范围内），没有则 None。"""
        for obj in self.world.objects:
            if self.distance_to(obj) <= obj.reach:
                return obj
        return None

    def look(self) -> dict:
        """给模型和前端共用的一份视图快照。"""
        near = self.nearby()
        here = self.at_object()
        return {
            "x": round(self.x, 2),
            "z": round(self.z, 2),
            "heading": round(self.heading, 1),
            "facing": self.facing_word(),
            "at": here.name if here else None,
            "target": self.target,
            "objects": [
                {
                    "name": o.name,
                    "x": o.x, "z": o.z,
                    "desc": o.desc,
                    "distance": round(d, 2),
                    "bearing": round(b, 1),
                    "direction": self.direction_word(b),
                    "reached": d <= o.reach,
                }
                for o, d, b in near
            ],
        }

    def describe(self) -> str:
        """自然语言描述，直接拼进提示词。

        数字给到 0.1 米：模型据此能判断"就在旁边"和"走过去要一会儿"，
        但不需要更精确——再细也只是噪声。
        """
        here = self.at_object()
        lines = []

        if here:
            lines.append(f"你在{here.name}旁。{here.desc}。")
        else:
            lines.append(f"你在房间中央偏 ({self.x:.1f}, {self.z:.1f}) 的位置，"
                         f"面向{self.facing_word()}。")

        if self.target:
            lines.append(f"你正在往{self.target}走，还没到。")

        near = [t for t in self.nearby() if t[0] is not here]
        if near:
            lines.append("你周围能看到的：")
            for obj, d, b in near[:4]:
                word = self.direction_word(b)
                if d <= obj.reach:
                    lines.append(f"  · {obj.name}：就在旁边。{obj.desc}")
                else:
                    lines.append(f"  · {obj.name}：{word}约 {d:.1f} 米。{obj.desc}")
        else:
            lines.append("周围空荡荡的，没有任何东西。")

        return "\n".join(lines)

    # ── 转身 ──
    def turn(self, direction: str = "around", degrees: float = 90.0) -> str:
        """原地转身，改变朝向。

        朝向会影响"某东西在我哪一侧"的措辞，进而影响模型描述自己
        在房间里的方式。转身不改变位置，所以价格很低，适合作为
        "我想确认一下"的轻量动作——比走一趟便宜得多。

        返回转身后的方位词，供上层写反馈。
        """
        if direction == "around":
            self.heading = (self.heading + 180.0) % 360
        elif direction == "left":
            self.heading = (self.heading - degrees) % 360
        elif direction == "right":
            self.heading = (self.heading + degrees) % 360
        return self.facing_word()

    # ── 移动 ──
    def move_toward(self, place: str, step: float = 1.0) -> Optional[str]:
        """朝目标走一步。

        返回 None 表示找不到这个目标；否则返回一个描述用的字符串
        （"接近了"/"到了"）。

        刻意只走一步而不是直接传送：移动应该是渐进的，走的过程中
        距离在变，模型下次看的时候能看到"更近了"。
        """
        obj = next((o for o in self.world.objects if o.name == place), None)
        if obj is None:
            return None

        self.target = place
        d = self.distance_to(obj)

        if d <= obj.reach:
            self.target = None
            return "arrived"

        # 朝目标的单位向量走一步，步长不超过剩余距离
        dx, dz = obj.x - self.x, obj.z - self.z
        travel = min(step, d)
        self.x += dx / d * travel
        self.z += dz / d * travel

        # 朝向对齐目标
        self.heading = math.degrees(math.atan2(dx, -dz)) % 360

        # 走出边界就拉回来，与前端 BOUND 的行为一致
        r = math.hypot(self.x, self.z)
        if r > self.world.radius:
            self.x *= self.world.radius / r
            self.z *= self.world.radius / r

        if self.distance_to(obj) <= obj.reach:
            self.target = None
            return "arrived"
        return "approaching"
