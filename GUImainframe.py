# M083040027 王鈞佑 Chun-Yu Wang 220201108
from volonoi import *
import tkinter as tk
from tkinter import filedialog
# def makeorder(list, stack):
#     if len(list) == 1:
#         stack.append(list)
#         return stack
#     Sl = []
#     Sr = []
#     median = np.mean(list, axis=0)
#     if list:
#         for point in list:
#             if point[0] <= median[0]:
#                 if point[0] < median[0]:
#                     Sl.append(point)
#                 else:
#                     if point[1] >= median[1]:
#                         Sl.append(point)
#                     else:
#                         Sr.append(point)
#             else:
#                 Sr.append(point)
#         stack = makeorder(Sl, stack)
#         stack = makeorder(Sr, stack)
#         stack.append(Sl + Sr)
#     return stack

class MainApplication():
    def __init__(self, master):
        if __name__ == '__main__':
            self.master = master
            master.title("M083040027 volonoi")
            master.geometry('1024x768')
            master.resizable(0, 0)
            self.guistr = tk.StringVar()
            self.var = tk.BooleanVar()
            self.startFlag = False
            self.stepFlag = False
            self.loadnextFlag = False
            self.pointlist = []
            self.listorder = []
            self.Inputdata = []
            self.count = 0
            self.inputnextcount = 0
            self.outputpointlist = []
            self.outputedgelist = []
            self.canvas = tk.Canvas(master, bg='#D0D0D0', height=600, width=600)
            self.canvas.bind('<Button 1>', lambda event: self.left_click(event))
            self.canvas.grid(row=0, column=0, rowspan=5)
            self.startBtn = tk.Button(self.master, text='start', width=15, height=2, command=lambda: self.start())
            self.startBtn.grid(row=0, column=1)
            self.StepStartBtn = tk.Button(self.master, text='stepstart', width=15, height=2,
                                          command=lambda : self.stepstart())
            self.StepStartBtn.grid(row=1, column=1)
            self.StepBtn = tk.Button(self.master, text='step', width=15, height=2, command=self.step)
            self.StepBtn.grid(row=1, column=2)
            self.cleanBtn = tk.Button(self.master, text='clear', width=15, height=2, command=lambda : self.clear())
            self.cleanBtn.grid(row=2, column=1)
            self.InputBtn = tk.Button(self.master, text='load', width=15, height=2, command=lambda: self.load())
            self.InputBtn.grid(row=3, column=1)
            self.OutputBtn = tk.Button(self.master, text='save', width=15, height=2, command=self.save)
            self.OutputBtn.grid(row=4, column=1)
            self.NextBtn = tk.Button(self.master, text='next', width=15, height=2,
                                     command=lambda: self.inputnext())
            self.NextBtn.grid(row=3, column=2)
            self.currentPointlist = tk.Label(self.master, text="Current PointList")
            self.currentPointlist.grid(row=5, column=0)

    def start(self):
        def takeSecond(elem):
            return elem[0]

        self.pointlist.sort(key=takeSecond)
        self.canvas.delete("all")
        self.printpoint(self.pointlist, 'black')
        if self.startFlag == False:
            self.guistr.set('press the button')
            if self.pointlist:
                points, edges = volonoialgorithm(self.pointlist, self)
                self.outputedgelist = edges
                self.outputpointlist = points
                self.drawResult(edges, points)
                self.stepFlag = False



    def clear(self):
        self.canvas.delete("all")
        self.startFlag = False
        self.stepFlag = False
        self.loadnextFlag = False
        self.pointlist = []
        self.listorder = []
        self.Inputdata = []
        self.count = 0
        self.inputnextcount = 0
        self.outputpointlist = []
        self.outputedgelist = []

    def load(self):
        self.loadnextFlag = True

        def RepresentsInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        self.master.filename = filedialog.askopenfilename(initialdir="./", parent=self.master, title='Choose a file')

        file = open(self.master.filename, encoding="UTF-8")
        lines = []
        self.Inputdata = []
        if file:
            for line in file:
                lines.append(line.rstrip('\n'))

            for i in range(len(lines)):
                if RepresentsInt(lines[i]):
                    point = []
                    if int(lines[i]) == 0:
                        break
                    for j in range(i + 1, i + int(lines[i]) + 1):
                        point.append(list(map(int, lines[j].split())))
                    self.Inputdata.append(point)
                else:
                    if lines[i] == "":
                        pass
                    elif lines[i][0] == "P":
                        newline = list(lines[i].split())
                        tmppoint = [float(newline[1]), float(newline[2])]
                        self.outputpointlist.append(tmppoint)
                        self.printpoint([tmppoint], 'black')
                    elif lines[i][0] == "E":
                        newline = list(lines[i].split())
                        tmpedge = [[float(newline[1]), float(newline[2])], [float(newline[3]), float(newline[4])]]
                        self.outputedgelist.append(tmpedge)
                        self.printedges([tmpedge], 'red', 'edge')
            for thelist in self.Inputdata:
                print(thelist)
                print("----------------")

            file.close()

    def save(self):
        outputfile = open('outputfile.txt', 'w')
        outputstr = ""
        for p in self.outputpointlist:
            outputstr += "P " + str(p[0]) + " " + str(p[1]) + "\n"
        for e in self.outputedgelist:
            outputstr += "E " + str(e[0][0]) + " " + str(e[0][1]) + " " + str(e[1][0]) + " " + str(e[1][1]) + "\n"
        outputfile.write(outputstr)
        outputfile.close()

    def inputnext(self):
        if self.loadnextFlag == True:
            if self.inputnextcount == len(self.Inputdata):
                print("the end")
                self.clear()
                self.inputnextcount = 0
                self.loadnextFlag = False
                self.currentPointlist.configure(text="Current PointList")
            elif self.inputnextcount >= 0:
                self.pointlist = self.Inputdata[self.inputnextcount]
                sumstr = ""
                for point in self.pointlist:
                    sumstr += "[%d, %d] " % (point[0], point[1])
                self.currentPointlist.configure(text=sumstr)

                self.inputnextcount += 1

    def step(self):
        if self.var.get():
            self.var.set(False)
        else:
            self.var.set(True)
        print(self.var.get())

    def stepstart(self):
        if not self.stepFlag:
            self.stepFlag = True
            self.start()

    def left_click(self, label):
        print('clicked left button at x = % d, y = % d' % (label.x, label.y))
        self.pointlist.append([label.x, label.y])
        self.printsinglepoint([label.x, label.y])
        for p in self.pointlist:
            print("pointlist: x = % d, y = % d" % (p[0], p[1]))

    def printsinglepoint(self, point):
        self.canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill='black', tags="point")

    def guiwait(self):
        print("waiting...")
        print(self.var.get())
        self.StepBtn.wait_variable(self.var)
        print("done waiting.")

    def printedges(self, edges, color, tag):
        for edge in edges:
            self.canvas.create_line(edge[0][0], edge[0][1], edge[1][0], edge[1][1], fill=color, tags=tag)

    def printpoint(self, points, color):
        for point in points:
            self.canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill=color, tags="point")

    def drawResult(self, edges, points):
        self.canvas.delete("all")
        self.printedges(edges, 'red', 'edge')
        self.printpoint(points, 'black')

if __name__ == '__main__':
    root = tk.Tk()
    myGUI = MainApplication(root)
    root.mainloop()
