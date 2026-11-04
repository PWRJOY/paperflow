"""
快递面单热敏打印机 —— 理线增高架 v3（极简）
================================================================
台面 80 x 130 mm（放 8x13cm 快递面单），台面高度 25 mm。
电源线 / USB 线直接盘在架子内部的敞开腔里，前后贯通、四面通透。

v3 相比 v2 的变化
  * 去掉侧壁上的 4 个弹性卡扣：腔体彻底干净，线随便盘，省料也省事
  * 结构就是「顶板 + 左右两片侧壁」，四面全通透（前、后、下都开着）

坐标系：X = 宽(80)，Y = 长(130)，Z = 高(25)
       原点在底面中心，Y+ 为"后"（有挡边一侧），Y- 为"前"（敞口便于抽取面单）

打印建议
  * 正放打印（顶板朝上），无需支撑，纯桥接。
  * 顶板桥接跨度 79.6 mm，请把桥接速度降到 25~30 mm/s、风扇开满；
    桥接面朝下藏在腔体里，不好看也无所谓。
  * PLA / PETG，3 层壁，20% 填充。

依赖: pip install build123d
"""

from build123d import (
    Axis, Box, Plane, Polyline, Pos,
    chamfer, export_step, export_stl, extrude, fillet, make_face,
)

# ============================================================
#  参数区（单位 mm，想改尺寸只动这里）
# ============================================================
W       = 80.0    # 台面宽度 X —— 对应面单 8cm
L       = 130.0   # 台面长度 Y —— 对应面单 13cm
H       = 25.0    # 台面高度 Z —— 对应增高 2.5cm

TOP_T   = 3.6     # 顶板厚（去掉底板和隔板后加厚一点补刚度）
WALL_T  = 3.2     # 侧壁厚
R_CORNER = 6.0    # 竖直棱圆角

LIP_H   = 4.0     # 顶面防滑挡边高（0 = 不做挡边）
LIP_T   = 3.0     # 挡边厚
TAPER_L = 45.0    # 前端挡边渐隐斜坡长度（0 = 不做斜坡）

# ============================================================
#  建模
# ============================================================
BODY_W = W + 2 * LIP_T
BODY_L = L + LIP_T
CY     = LIP_T / 2.0

# --- 1. 外形柱体 + 竖直棱圆角 ---------------------------------
outer = Pos(0, CY, H / 2) * Box(BODY_W, BODY_L, H)
outer = fillet(outer.edges().filter_by(Axis.Z), R_CORNER)

# --- 2. 掏空内腔（底部全开、前后贯通）--------------------------
CAV_H = H - TOP_T
cavity = Pos(0, CY, CAV_H / 2) * Box(
    BODY_W - 2 * WALL_T, BODY_L + 6, CAV_H
)
body = outer - cavity

# --- 3. 顶面 U 形防滑挡边（后 + 左 + 右，前面敞开）-------------
if LIP_H > 0:
    lip_outer = Pos(0, CY, H + LIP_H / 2) * Box(BODY_W, BODY_L, LIP_H)
    lip_outer = fillet(lip_outer.edges().filter_by(Axis.Z), R_CORNER)
    # 内框向前穿出 -> 前面自然没有挡边
    lip_inner = Pos(0, -1.0, H + LIP_H / 2) * Box(W, L + 4, LIP_H * 3)
    body = body + (lip_outer - lip_inner)

    # 挡边顶面圆角，手感不割手
    top_edges = body.edges().filter_by(Plane.XY).group_by(Axis.Z)[-1]
    body = fillet(top_edges, 0.4)

    # 前段挡边渐隐成斜坡：正面零高度 -> 后段全高，推面单不刮手
    if TAPER_L > 0:
        y0 = -L / 2
        y1 = y0 + TAPER_L
        sec = Plane.YZ * make_face(
            Polyline(
                (y0 - 1, H), (y0 - 1, H + LIP_H + 12),
                (y1, H + LIP_H + 12), (y1, H + LIP_H),
                close=True,
            )
        )
        body = body - extrude(sec, amount=BODY_W, both=True)

# --- 4. 底面一圈小倒角，抑制象脚 ------------------------------
bottom_edges = body.edges().filter_by(Plane.XY).group_by(Axis.Z)[0]
body = chamfer(bottom_edges, 0.3)

part = body

# ============================================================
#  导出
# ============================================================
if __name__ == "__main__":
    from pathlib import Path
    # 输出固定到项目根目录的 output/，与从哪个目录运行无关
    OUT = Path(__file__).resolve().parent.parent / "output"
    OUT.mkdir(parents=True, exist_ok=True)

    export_step(part, str(OUT / "riser.step"))
    export_stl(part, str(OUT / "riser.stl"), tolerance=0.05, angular_tolerance=0.25)

    bb = part.bounding_box()
    print("外廓尺寸 (mm): "
          f"X {bb.size.X:.2f} x Y {bb.size.Y:.2f} x Z {bb.size.Z:.2f}")
    print("体积 (cm^3) :", round(part.volume / 1000, 1))
    print("实体数量    :", len(part.solids()))
