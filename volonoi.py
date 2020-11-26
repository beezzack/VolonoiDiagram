# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

screensize = 800

# Constant integers for directions
RIGHT = 1
LEFT = -1
ZERO = 0

# Constant integers for triangle
WIDE_ANGLE = 1
ACUTE_ANGLE = -1
RIGHT_ANGLE = 0


class Vp:
    def __init__(self, InputP):
        # polygon = inPoint+1, Vertex = 2*inPoint - 2, Edge = 3*inPoint - 3

        # polygon: inPoint+1
        self.edgeAoundPolygon = []
        # polygon point
        self.xPolygon = [99999]
        self.yPolygon = [99999]

        # Vertex: 2*inPoint - 2
        self.edgeAoundVetex = []

        # Edge: 3*inPoint - 3
        self.rightPloygon = []
        self.leftPloygon = []
        self.startVertex = []
        self.endVertex = []
        # cw: clockwise, ccw: counterclockwise
        self.cwPredecessor = []
        self.ccwPredecessor = []
        self.cwSuccessor = []
        self.ccwSuccessor = []

        # wVertex: 1 if vertex is ordinary point, 0 if vertex at infinity
        self.xVertex = []
        self.yVertex = []
        self.wVertex = []

        for point in InputP:
            self.xPolygon.insert(0, point[0])
            self.yPolygon.insert(0, point[1])

    def getedgesize(self):
        return len(self.startVertex)

    def getPolysize(self):
        return len(self.xPolygon)

    def getVersize(self):
        return len(self.xVertex)

    def getverIndex(self, point):
        for i in range(self.getVersize()):
            if self.xVertex[i] == point[0] and self.yVertex[i] == point[1]:
                return i
        return False

    def findP(self, point):
        for i in range(len(self.xPolygon)):
            if point[0] == self.xPolygon[i] and point[1] == self.yPolygon[i]:
                return i
        return False


def getEdgePolygon(theVp, vertexNum):
    # Input : VP, integer
    # output: Lc of edges and Lr of polygon that surround vertex j ccw.
    Lc = []
    Lr = []
    k = theVp.edgeAoundVetex[vertexNum]
    kstart = k
    while 1:
        Lc.append(k)
        if vertexNum == theVp.startVertex[k]:
            Lr.append(theVp.leftPloygon[k])
            k = theVp.ccwPredecessor[k]
        else:
            Lr.append(theVp.rightPloygon[k])
            k = theVp.ccwSuccessor[k]

        if k == kstart:
            return Lc, Lr


def getEdgeVertex(theVp, PolygonNum):
    # Input : VP, integer
    # output: Lc of edges and Lv of vertex that surround polygon i cw.
    Lc = []
    Lv = []
    k = theVp.edgeAoundPolygon[PolygonNum]
    kstart = k
    while 1:
        Lc.append(k)
        if PolygonNum == theVp.leftPolygon[k]:
            Lv.append(theVp.endVertex[k])
            k = theVp.cwSuccessor[k]
        else:
            Lv.append(theVp.startVertex[k])
            k = theVp.cwPredecessor[k]

        if k == kstart:
            return Lc, Lv


def getHull(VD):
    # Input VD
    # Ouput list[list]
    global polygon
    convexhull = []
    versize = VD.getVersize()
    if versize <= 3:
        if versize == 0:
            convexhull.append([VD.xPolygon[0], VD.yPolygon[0]])
        else:
            for i in range(versize):
                convexhull.append([VD.xPolygon[i], VD.yPolygon[i]])
        return convexhull

    for i in range(len(VD.startVertex)):
        startV = VD.startVertex[i]
        endV = VD.endVertex[i]
        if (VD.wVertex[startV] == 1 and VD.wVertex[endV] == 0) or (VD.wVertex[startV] == 0 and VD.wVertex[endV] == 1):
            # an infinite ray
            polygon = VD.leftPloygon[i]
            convexhull.append([VD.xPolygon[polygon], VD.yPolygon[polygon]])
            break

    startploygon = polygon
    while 1:
        Lc, Lv = getEdgeVertex(VD, startploygon)
        for edge in Lc:
            startV = VD.startVertex[edge]
            endV = VD.endVertex[edge]
            if (VD.wVertex[startV] == 1 and VD.wVertex[endV] == 0) or (
                    VD.wVertex[startV] == 0 and VD.wVertex[endV] == 1):
                # an infinite ray
                startploygon = VD.leftPloygon[edge]
                convexhull.append([VD.xPolygon[startploygon], VD.yPolygon[startploygon]])
                break
        if polygon == startploygon:
            break

    return convexhull


def directionOfPoint(A, B, P):
    global RIGHT, LEFT, ZERO

    # Subtracting co-ordinates of
    # point A from B and P, to
    # make A as origin
    vectorAP = [P[0] - A[0], P[1] - A[1]]
    vectorBP = [P[0] - B[0], P[1] - B[1]]

    def cross_product(p1, p2):
        result = p1[0] * p2[1] - p2[0] * p1[1]
        if abs(result) < 0.01:
            result = 0
        return result

    # Determining cross Product
    cross_product_result = cross_product(vectorAP, vectorBP)

    # Return RIGHT if cross product is positive
    if cross_product_result > 0:
        return RIGHT

    # Return LEFT if cross product is negative
    if cross_product_result < 0:
        return LEFT

    # Return ZERO if cross product is zero
    return ZERO


# checks if p3 makes left turn at p2
def left(p1, p2, p3):
    return directionOfPoint(p1, p2, p3) < 0


# checks if p3 makes right turn at p2
def right(p1, p2, p3):
    return directionOfPoint(p1, p2, p3) > 0
    # checks if p1, p2 and p3 are collinear


def collinear(p1, p2, p3):
    return directionOfPoint(p1, p2, p3) == 0


def getTangent(SlHull, SrHull):
    # Input : list[list], list[list]
    # Output: list[list], list[list]
    Leftpoint = SlHull[0]
    Rightpoint = SrHull[0]
    l = 0
    r = 0

    for point in SlHull:
        if point[0] > Leftpoint[0]:
            Leftpoint = point
            l = SlHull.index(Leftpoint)
        elif point[0] == Leftpoint[0]:
            if point[1] < Leftpoint[1]:
                Leftpoint = point
                l = SlHull.index(Leftpoint)

    for point in SrHull:
        if point[0] < Rightpoint[0]:
            Rightpoint = point
            r = SrHull.index(Rightpoint)
        elif point[0] == Rightpoint[0]:
            if point[1] > Rightpoint[1]:
                Rightpoint = point
                r = SrHull.index(Rightpoint)

    lpointer = l
    rpointer = r
    prevlpointer = None
    prevrpointer = None
    while True:
        prevlpointer = lpointer
        prevrpointer = rpointer
        while left(SlHull[lpointer], SrHull[rpointer], SrHull[(rpointer + 1) % len(SrHull)]):
            rpointer = (rpointer + 1) % len(SrHull)
        # move p as long as it makes right turn
        while right(SrHull[rpointer], SlHull[lpointer], SlHull[(lpointer - 1) % len(SlHull)]):
            lpointer = (lpointer - 1) % len(SlHull)
        if lpointer == prevlpointer and rpointer == prevrpointer:
            break
    while True:
        prevlpointer = l
        prevrpointer = r
        while right(SlHull[l], SrHull[r], SrHull[(r - 1) % len(SrHull)]):
            r = (r - 1) % len(SrHull)
        # move p as long as it makes right turn
        while left(SrHull[r], SlHull[l], SlHull[(l + 1) % len(SlHull)]):
            l = (l + 1) % len(SlHull)
        if l == prevlpointer and r == prevrpointer:
            break

    Lowtengent = [SlHull[lpointer], SrHull[rpointer]]
    Hightengent = [SlHull[l], SrHull[r]]
    return Hightengent, Lowtengent


def getBS(sg):
    # Input list[list], a line
    # output list[list], a line
    X = (sg[1][0] - sg[0][0])
    Y = (sg[1][1] - sg[0][1])
    BSvector = [-Y, X]
    mid = [(sg[1][0] + sg[0][0]) / 2, (sg[1][1] + sg[0][1]) / 2]

    BS = [[0, 0], [0, 0]]
    # BS 垂直
    if BSvector[0] == 0:
        BS[0] = [mid[0], screensize]
        BS[1] = [mid[0], 0]
    # BS 平行
    elif BSvector[1] == 0:
        BS[0] = [screensize, mid[1]]
        BS[1] = [0, mid[1]]
    else:
        m1 = (0 - mid[1]) / BSvector[1]
        m2 = (screensize - mid[1]) / BSvector[1]
        # BS[0][0] = mid[0] + m1 * BSvector[0]
        # BS[0][1] = 0
        BS[0][0] = mid[0] + m2 * BSvector[0]
        BS[0][1] = screensize
        BS[1][0] = mid[0] + m1 * BSvector[0]
        BS[1][1] = 0

    return BS


def unionhp_Tri(sg, hp, sgBS, Sl, Sr):
    if not hp:
        hp.append(sgBS)
        return hp
    points = Sl + Sr
    category = ACUTE_ANGLE
    mid = [(sg[1][0] + sg[0][0]) / 2, (sg[1][1] + sg[0][1]) / 2]

    def getAngle(P1, P2, P3):
        a = np.array(P1)
        b = np.array(P2)
        c = np.array(P3)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        result = np.degrees(angle)
        print(result)
        return result
        # ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        # return ang + 360 if ang < 0 else ang

    def getThirdpoint(sg, points):
        for point in points:
            if sg[0] != point and sg[1] != point:
                return point

    A = points[0]
    B = points[1]
    C = points[2]
    angles = []
    angles.append(getAngle(A, B, C))
    angles.append(getAngle(A, C, B))
    angles.append(getAngle(B, A, C))
    lines = [[[A[0], A[1]], [C[0], C[1]]], [[A[0], A[1]], [B[0], B[1]]], [[B[0], B[1]], [C[0], C[1]]]]
    # last point
    L1 = line(hp[0][0], hp[0][1])
    L2 = line(sgBS[0], sgBS[1])
    R = intersection(L1, L2)
    angleindex = 0
    for angle in angles:
        if angle > 90:
            angleindex = angles.index(angle)
            category = WIDE_ANGLE
        elif angle == 90:
            angleindex = angles.index(angle)
            category = RIGHT_ANGLE

    thelongestline = lines[angleindex]
    if category == WIDE_ANGLE:
        # 4 line hp[0][0] -> R -> BS[1], hp[0][0] -> R -> BS[0], BS[1]-> R -> hp[0][1], BS[1]-> R -> hp[0][1],
        if sg == thelongestline:
            if insidetheline(sgBS[0], R, mid):
                hp[0][1] = R
                hp.append([[R[0], R[1]], sgBS[1]])
            elif insidetheline(sgBS[1], R, mid):
                tmp = hp[0]
                hp[0][0] = sgBS[0]
                hp[0][1] = R
                hp.append([[R[0], R[1]], tmp[1]])
        else:
            hp[0][1] = [R[0], R[1]]
            if insidetheline(sgBS[0], R, mid):
                hp.append([[R[0], R[1]], sgBS[0]])
            else:
                hp.append([[R[0], R[1]], sgBS[1]])

    elif category == RIGHT_ANGLE:
        # 2 line hp[0][0] -> R -> BS[1], BS[0]-> R -> hp[0][1]
        if sg == thelongestline:
            thirdpoint = getThirdpoint(sg, points)
            if directionOfPoint(sg[0], sg[1], thirdpoint) == directionOfPoint(sg[0], sg[1], sgBS[0]):
                hp[0][1] = [R[0], R[1]]
                hp.append([[R[0], R[1]], sgBS[1]])
            else:
                tmp = hp[0]
                hp[0][0] = sgBS[0]
                hp[0][1] = [R[0], R[1]]
                hp.append([[R[0], R[1]], tmp[1]])
        else:
            hp[0][1] = [R[0], R[1]]
            if insidetheline(sgBS[0], R, mid):
                hp.append([[R[0], R[1]], sgBS[0]])
            else:
                hp.append([[R[0], R[1]], sgBS[1]])


    else:
        hp[0][1] = [R[0], R[1]]
        if insidetheline(sgBS[0], R, mid):
            hp.append([[R[0], R[1]], sgBS[0]])
        else:
            hp.append([[R[0], R[1]], sgBS[1]])

    return hp


def unionhp(sg, hp, sgBS):
    # Input list[line], line = [[x1, y1], [x2, y2]] 
    # output list[line]
    mid = [(sg[1][0] + sg[0][0]) / 2, (sg[1][1] + sg[0][1]) / 2]
    # 2point
    if not hp:
        hp.append(sgBS)
    else:
        lastpointindex = len(hp) - 1
        # last point
        L1 = line(hp[lastpointindex][0], hp[lastpointindex][1])
        L2 = line(sgBS[0], sgBS[1])
        R = intersection(L1, L2)

        hp[lastpointindex][1] = [R[0], R[1]]
        hp.append([[R[0], R[1]], sgBS[1]])

    return hp


def line(p1, p2):
    # point point 
    # ABC
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x, y]
    else:
        print(str(L1[0]) + "," + str(L1[1]) + " and " + str(L2[0]) + "," + str(L2[1]) + " no R.")
        return [-1, -1]


def insidetheline(startpoint, endpoint, param):
    if startpoint[0] == endpoint[0]:
        if startpoint[0] - param[0] > 0.001:
            return False
        else:
            if min(startpoint[1], endpoint[1]) <= param[1] <= max(startpoint[1], endpoint[1]):
                return True
            else:
                return False

    if startpoint[1] == endpoint[1]:
        if startpoint[1] - param[1] > 0.001:
            return False
        else:
            if min(startpoint[0], endpoint[0]) <= param[0] <= max(startpoint[0], endpoint[0]):
                return True
            else:
                return False

    if min(startpoint[0], endpoint[0]) <= param[0] <= max(startpoint[0], endpoint[0]) and min(startpoint[1],
                                                                                              endpoint[1]) <= param[
        1] <= max(startpoint[1], endpoint[1]):
        return True
    else:
        return False


def findvertexindex(point, VD):
    size = VD.getVersize()
    for i in range(size):
        if VD.xVertex[i] == point[0] and VD.yVertex[i] == point[1]:
            return i


def findpolygonindex(poly, SlVD):
    index = 0
    for i in range(SlVD.getPolysize()):
        if SlVD.xPolygon[i] == poly[0] and SlVD.yPolygon[i] == poly[1]:
            index = i
    return index


def discardVD(hp, VD, myGUI):
    hpsize = len(hp)
    hpset = [hp[0][0]]
    for i in range(hpsize):
        hpset.append(hp[i][1])

    if VD.startVertex and VD.endVertex:
        for i in range(VD.getedgesize()):
            start = VD.startVertex[i]
            end = VD.endVertex[i]
            # start vertex index
            startpoint = [VD.xVertex[start], VD.yVertex[start]]
            endpoint = [VD.xVertex[end], VD.yVertex[end]]
            for j in range(len(hpset)):
                if collinear(startpoint, endpoint, hpset[j]) and insidetheline(startpoint, endpoint,
                                                                                       hpset[j]):
                    startdirection = directionOfPoint([hpset[j-1][0], hpset[j-1][1]], [hpset[j][0], hpset[j][1]], startpoint)
                    hpdirection = directionOfPoint([hpset[j-1][0], hpset[j-1][1]], [hpset[j][0], hpset[j][1]], [hpset[j+1][0], hpset[j+1][1]])
                    if hpdirection == LEFT:
                        if startdirection == RIGHT:
                            VD.xVertex[end] = hpset[j][0]
                            VD.yVertex[end] = hpset[j][1]
                            VD.ccwSuccessor[i] = 0
                            VD.cwSuccessor[i] = 0
                        else:
                            VD.xVertex[start] = hpset[j][0]
                            VD.yVertex[start] = hpset[j][1]
                            VD.ccwPredecessor[i] = 0
                            VD.cwPredecessor[i] = 0
                    elif hpdirection == RIGHT:
                        if startdirection == LEFT:
                            VD.xVertex[end] = hpset[j][0]
                            VD.yVertex[end] = hpset[j][1]
                            VD.ccwSuccessor[i] = 0
                            VD.cwSuccessor[i] = 0
                        else:
                            VD.xVertex[start] = hpset[j][0]
                            VD.yVertex[start] = hpset[j][1]
                            VD.ccwPredecessor[i] = 0
                            VD.cwPredecessor[i] = 0


                    # if direction == LEFT:
                    #     if directionOfPoint(hpset[0], hpset[hpsize - 1], midpoint) == LEFT:
                    #         VD.xVertex[start] = hppoint[0]
                    #         VD.yVertex[start] = hppoint[1]
                    #         VD.ccwPredecessor[i] = 0
                    #         VD.cwPredecessor[i] = 0
                    #     else:
                    #         VD.xVertex[end] = hppoint[0]
                    #         VD.yVertex[end] = hppoint[1]
                    #         VD.ccwSuccessor[i] = 0
                    #         VD.cwSuccessor[i] = 0
                    # elif direction == RIGHT:
                    #     if directionOfPoint(hpset[0], hpset[hpsize - 1], midpoint) == RIGHT:
                    #         VD.xVertex[start] = hppoint[0]
                    #         VD.yVertex[start] = hppoint[1]
                    #         VD.ccwPredecessor[i] = 0
                    #         VD.cwPredecessor[i] = 0
                    #     else:
                    #         VD.xVertex[end] = hppoint[0]
                    #         VD.yVertex[end] = hppoint[1]
                    #         VD.ccwSuccessor[i] = 0
                    #         VD.cwSuccessor[i] = 0
            myGUI.printedges([[[VD.xVertex[start], VD.yVertex[start]], [VD.xVertex[end], VD.yVertex[end]]]], 'blue',
                             'discardresult')
            if myGUI.stepFlag:
                myGUI.guiwait()
    myGUI.canvas.delete('discardresult')
    return VD


def reconstruct(hp, SlVD, SrVD, sgpolygen, myGUI):
    hpsize = len(hp)
    if hpsize > 1:
        SlVD = discardVD(hp, SlVD, myGUI)
        SrVD = discardVD(hp, SrVD, myGUI)

    SlVDedge = SlVD.getedgesize()
    SlVDpoly = SlVD.getPolysize()
    SlVDver = SlVD.getVersize()
    SrVDedge = SrVD.getedgesize()
    SrVDpoly = SrVD.getPolysize()
    SrVDver = SrVD.getVersize()
    # input right vertex in to left
    for i in range(SrVDver):
        SlVD.xVertex.append(SrVD.xVertex[i])
        SlVD.yVertex.append(SrVD.yVertex[i])
        SlVD.wVertex.append(SrVD.wVertex[i])
    # input right polygon in to left
    for i in range(SrVDpoly):
        # polygon: inPoint+1
        if SrVD.edgeAoundPolygon:
            SlVD.edgeAoundPolygon.append(SrVD.edgeAoundPolygon[i] + SlVDedge)
        # polygon point
        SlVD.xPolygon.append(SrVD.xPolygon[i])
        SlVD.yPolygon.append(SrVD.yPolygon[i])
    SlVD.xPolygon.pop(SlVDpoly - 1)
    SlVD.yPolygon.pop(SlVDpoly - 1)
    newinfiniteindex = SlVD.getPolysize() - 1
    # 先處理無限邊
    for i in range(SlVDedge):
        # Edge: 3*inPoint - 3
        if SlVD.rightPloygon[i] == SlVDpoly - 1:
            SlVD.rightPloygon[i] = newinfiniteindex
        elif SlVD.leftPloygon[i] == SlVDpoly - 1:
            SlVD.leftPloygon[i] = newinfiniteindex
    # input right edge in to left
    for i in range(SrVDedge):
        SlVD.rightPloygon.append(SrVD.rightPloygon[i] + SlVDpoly - 1)
        SlVD.leftPloygon.append(SrVD.leftPloygon[i] + SlVDpoly - 1)

        SlVD.startVertex.append(SrVD.startVertex[i] + SlVDver)
        SlVD.endVertex.append(SrVD.endVertex[i] + SlVDver)

        # cw: clockwise, ccw: counterclockwise
        SlVD.cwPredecessor.append(SrVD.cwPredecessor[i] + SlVDedge)
        SlVD.ccwPredecessor.append(SrVD.ccwPredecessor[i] + SlVDedge)
        SlVD.cwSuccessor.append(SrVD.cwSuccessor[i] + SlVDedge)
        SlVD.ccwSuccessor.append(SrVD.ccwSuccessor[i] + SlVDedge)

    for i in range(hpsize):
        leftindex = findpolygonindex(sgpolygen[i][0], SlVD)
        rightindex = findpolygonindex(sgpolygen[i][1], SlVD)
        SlVD.leftPloygon.append(leftindex)
        SlVD.rightPloygon.append(rightindex)

        if i == 0:
            SlVD.xVertex.append(hp[i][0][0])
            SlVD.yVertex.append(hp[i][0][1])
            SlVD.wVertex.append(0)

            SlVD.xVertex.append(hp[i][1][0])
            SlVD.yVertex.append(hp[i][1][1])
            if hpsize == 1:
                SlVD.wVertex.append(0)
            else:
                SlVD.wVertex.append(1)
        elif i == hpsize - 1:
            SlVD.xVertex.append(hp[i][1][0])
            SlVD.yVertex.append(hp[i][1][1])
            SlVD.wVertex.append(0)

        else:
            SlVD.xVertex.append(hp[i][1][0])
            SlVD.yVertex.append(hp[i][1][1])
            SlVD.wVertex.append(1)

        SlVD.startVertex.append(findvertexindex(hp[i][0], SlVD))
        SlVD.endVertex.append(findvertexindex(hp[i][1], SlVD))
        # cw: clockwise, ccw: counterclockwise
        SlVD.cwPredecessor.append(0)
        SlVD.ccwPredecessor.append(0)
        SlVD.cwSuccessor.append(0)
        SlVD.ccwSuccessor.append(0)

    for i in range(SlVD.getedgesize()):
        # cw: clockwise, ccw: counterclockwise

        startpoint = SlVD.startVertex[i]
        endpoint = SlVD.endVertex[i]
        startlist = []
        endlist = []

        for j in range(SlVD.getedgesize()):
            if SlVD.startVertex[i] == startpoint or SlVD.endVertex[i] == startpoint:
                startlist.append([SlVD.startVertex[i], SlVD.endVertex[i]])
            elif SlVD.startVertex[i] == endpoint or SlVD.endVertex[i] == endpoint:
                endlist.append([SlVD.startVertex[i], SlVD.endVertex[i]])

        for point in startlist:
            if point[1] == endpoint or point[0] == endpoint:
                SlVD.cwPredecessor[i] = endpoint
                SlVD.ccwPredecessor[i] = endpoint
            elif point[1] != endpoint:
                direction = directionOfPoint([SlVD.xVertex[startpoint], SlVD.yVertex[startpoint]],
                                             [SlVD.xVertex[endpoint], SlVD.yVertex[endpoint]],
                                             [SlVD.xVertex[point[1]], SlVD.yVertex[point[1]]])
                if direction == RIGHT:
                    SlVD.cwPredecessor[i] = point[1]
                elif direction == LEFT:
                    SlVD.ccwPredecessor[i] = point[1]
            elif point[0] != endpoint:
                direction = directionOfPoint([SlVD.xVertex[startpoint], SlVD.yVertex[startpoint]],
                                             [SlVD.xVertex[endpoint], SlVD.yVertex[endpoint]],
                                             [SlVD.xVertex[point[0]], SlVD.yVertex[point[0]]])
                if direction == RIGHT:
                    SlVD.cwPredecessor[i] = point[0]
                elif direction == LEFT:
                    SlVD.ccwPredecessor[i] = point[0]

        for point in endlist:
            if point[1] == startpoint or point[0] == startpoint:
                SlVD.cwSuccessor[i] = startpoint
                SlVD.ccwSuccessor[i] = startpoint
            elif point[1] != startpoint:
                direction = directionOfPoint([SlVD.xVertex[startpoint], SlVD.yVertex[startpoint]],
                                             [SlVD.xVertex[endpoint], SlVD.yVertex[endpoint]],
                                             [SlVD.xVertex[point[1]], SlVD.yVertex[point[1]]])
                if direction == RIGHT:
                    SlVD.ccwSuccessor[i] = point[1]
                elif direction == LEFT:
                    SlVD.cwSuccessor[i] = point[1]
            elif point[0] != startpoint:
                direction = directionOfPoint([SlVD.xVertex[startpoint], SlVD.yVertex[startpoint]],
                                             [SlVD.xVertex[endpoint], SlVD.yVertex[endpoint]],
                                             [SlVD.xVertex[point[0]], SlVD.yVertex[point[0]]])
                if direction == RIGHT:
                    SlVD.ccwSuccessor[i] = point[0]
                elif direction == LEFT:
                    SlVD.cwSuccessor[i] = point[0]
    return SlVD


def printhp(hp, myGUI):
    myGUI.canvas.delete('hp')
    edges = []
    for h in hp:
        edges.append([(h[0][0], h[0][1]), (h[1][0], h[1][1])])

    myGUI.printedges(edges, 'red', 'hp')


def merageVD(Sl, Sr, SlVD, SrVD, myGUI):
    # Input: list, list, VD, VD
    # Output: VD
    # step 1:
    # 回傳座標值
    SlHull = getHull(SlVD)
    SrHull = getHull(SrVD)
    # step 2
    hightangent, lowtangent = getTangent(SlHull, SrHull)
    # step 3
    SlHightangent = hightangent[0]
    SrHightangent = hightangent[1]
    SlLowtangent = lowtangent[0]
    Srlowtangent = lowtangent[1]
    hightangentindex = [SlHull.index(SlHightangent), SrHull.index(SrHightangent)]
    lowtangentindex = [SlHull.index(SlLowtangent), SrHull.index(Srlowtangent)]
    sg = [SlHightangent, SrHightangent]
    sgpolygen = []
    hp = []
    # sl走訪順序
    front = hightangentindex[0]
    rear = lowtangentindex[0]
    Slseq = [SlHull[hightangentindex[0]]]
    if len(SlHull) > 1:
        while True:
            front = (front - 1) % len(SlHull)
            Slseq.append(SlHull[front])
            if front == rear:
                break
    # sr走訪順序
    front = hightangentindex[1]
    rear = lowtangentindex[1]
    Srseq = [SrHull[hightangentindex[1]]]
    if len(SrHull) > 1:
        while True:
            front = (front + 1) % len(SrHull)
            Srseq.append(SrHull[front])
            if front == rear:
                break
    Slpointer = 0
    Srpointer = 0
    while 1:
        Slnextindex = None
        Srnextindex = None
        sgpolygen.append(sg)
        lR = [-1, -1]
        rR = [-1, -1]
        sgBS = getBS(sg)
        pointlen = len(Sl) + len(Sr)
        if pointlen == 2:
            hp = unionhp(sg, hp, sgBS)
        elif pointlen == 3:
            hp = unionhp_Tri(sg, hp, sgBS, Sl, Sr)
        else:
            hp = unionhp(sg, hp, sgBS)

        printhp(hp, myGUI)
        if myGUI.stepFlag:
            myGUI.guiwait()
        if sg == lowtangent:
            break

        # step 4
        # Z是下一個可能的convex hull point
        # 1:1

        # if SlZindex != lowtangentindex[0]:
        #     Slnextindex = (SlZindex - 1) % len(SlHull)
        # if SrZindex != lowtangentindex[1]:
        #     Srnextindex = (SrZindex + 1) % len(SrHull)

        if Slpointer + 1 < len(Slseq):
            Slnextindex = Slpointer + 1
        if Srpointer + 1 < len(Srseq):
            Srnextindex = Srpointer + 1
        if Srnextindex is not None:
            newBS = getBS([sg[1], Srseq[Srnextindex]])
            rR = intersection(line(sgBS[0], sgBS[1]), line(newBS[0], newBS[1]))
            if abs(rR[0] - hp[len(hp) - 1][0][0]) < 0.01 and abs(rR[1] - hp[len(hp) - 1][0][1]) < 0.01:
                rR = [-1, -1]
        if Slnextindex is not None:
            newBS = getBS([sg[0], Slseq[Slnextindex]])
            lR = intersection(line(sgBS[0], sgBS[1]), line(newBS[0], newBS[1]))
            if abs(lR[0] - hp[len(hp) - 1][0][0]) < 0.01 and abs(lR[1] - hp[len(hp) - 1][0][1]) < 0.01:
                lR = [-1, -1]

        # 只有一個焦點
        if lR == [-1, -1] and rR != lR:
            sg = [sg[0], Srseq[Srnextindex]]
            Srpointer += 1
        elif rR == [-1, -1] and lR != rR:
            sg = [Slseq[Slnextindex], sg[1]]
            Slpointer += 1
        else:
            if lR[1] < rR[1]:
                sg = [sg[0], Srseq[Srnextindex]]
                Srpointer += 1
            else:
                sg = [Slseq[Slnextindex], sg[1]]
                Slpointer += 1
    # step 5
    resultVD = reconstruct(hp, SlVD, SrVD, sgpolygen, myGUI)
    return resultVD


def printVD(pointlist, VD, color, myGUI):
    edges = []
    for i in range(VD.getedgesize()):
        startX = VD.xVertex[VD.startVertex[i]]
        startY = VD.yVertex[VD.startVertex[i]]
        endX = VD.xVertex[VD.endVertex[i]]
        endY = VD.yVertex[VD.endVertex[i]]
        edges.append([(startX, startY), (endX, endY)])

    edges = sorted(edges, key=lambda edge: [edge[0][0], edge[0][1], edge[1][0], edge[1][1]])
    myGUI.printedges(edges, 'black', 'edge')
    myGUI.printpoint(pointlist, color)


def clearVD(myGUI):
    myGUI.canvas.itemconfig("point", fill='black')


def getVolonoi(inputPoints, myGUI):
    # Input: list[list]
    # Output: VP
    global Btn
    Sl = []
    Sr = []
    if len(inputPoints) == 1:
        return Vp(inputPoints)
    median = np.median(inputPoints, axis=0)
    # print(median)
    for point in inputPoints:
        if point[0] <= median[0]:
            if point[0] < median[0]:
                Sl.append(point)
            else:
                if point[1] >= median[1]:
                    Sl.append(point)
                else:
                    Sr.append(point)
        else:
            Sr.append(point)
    SlVD = getVolonoi(Sl, myGUI)
    SrVD = getVolonoi(Sr, myGUI)

    clearVD(myGUI)
    printVD(Sl, SlVD, 'blue', myGUI)
    printVD(Sr, SrVD, 'yellow', myGUI)
    if myGUI.stepFlag:
        myGUI.guiwait()
    ResultVD = merageVD(Sl, Sr, SlVD, SrVD, myGUI)

    return ResultVD


# Press the green button in the gutter to run the script.

def volonoialgorithm(pointlist, myGUI):
    cleanlist = []
    arr = [[200, 200], [200, 300], [200, 400]]
    # pointlist 處理
    pointset = set(map(tuple, pointlist))
    for s in pointset:
        newlist = [s[0], s[1]]
        cleanlist.append(newlist)
    pointlist = cleanlist
    pointlist = sorted(pointlist, key=lambda k: [k[0], k[1]])

    edges = []
    result = getVolonoi(pointlist, myGUI)

    for i in range(result.getedgesize()):
        startX = result.xVertex[result.startVertex[i]]
        startY = result.yVertex[result.startVertex[i]]
        endX = result.xVertex[result.endVertex[i]]
        endY = result.yVertex[result.endVertex[i]]
        edges.append([(startX, startY), (endX, endY)])

    edges = sorted(edges, key=lambda edge: [edge[0][0], edge[0][1], edge[1][0], edge[1][1]])
    for i in range(result.getedgesize()):
        print("E: (" + str(edges[i][0][0]) + ", " + str(edges[i][0][1]) + "), (" + str(edges[i][1][0]) + ", " + str(
            edges[i][1][1]) + ")")
    return pointlist, edges
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
