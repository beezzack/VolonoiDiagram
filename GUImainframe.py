#M083040027 王鈞佑 Chun-Yu Wang 220201108
import codecs


import tkinter as tk
from tkinter import filedialog

import numpy as np

import volonoi as vp

if __name__ == '__main__':

    window = tk.Tk()
    window.title('my window')
    window.geometry('1024x768')
    window.resizable(0,0)
    var = tk.StringVar()
    startFlag = False
    stepFlag = False
    loadnextFlag = False
    pointlist = []
    listorder = []
    Inputdata = []
    count = 0
    inputnextcount = 0

    outputpointlist = []
    outputedgelist = []


    def printsinglepoint(point):
        canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill='black')


    def left_click(label):
        global pointlist
        print('clicked left button at x = % d, y = % d' % (label.x, label.y))
        pointlist.append([label.x, label.y])
        printsinglepoint([label.x, label.y])
        for p in pointlist:
            print("pointlist: x = % d, y = % d" % (p[0], p[1]))


    def printedges(edges, color):
        for edge in edges:
            canvas.create_line(edge[0][0], edge[0][1], edge[1][0], edge[1][1], fill=color)
        #canvas.update()


    def printpoint(points, color):
        for point in points:
            canvas.create_oval(point[0]-2, point[1]-2, point[0]+2, point[1]+2, fill=color)




    def drawResult(edges, points):
        canvas.delete("all")
        printedges(edges, 'red')
        printpoint(points, 'black')


    def start():
        global startFlag
        global pointlist
        global outputpointlist
        global outputedgelist

        def takeSecond(elem):
            return elem[0]
        pointlist.sort(key=takeSecond)
        if startFlag == False:
            var.set('press the button')
            if pointlist:
                points, edges = vp.volonoialgorithm(pointlist)
                outputedgelist = edges
                outputpointlist = points
                drawResult(edges, points)

        # else:
        #     startFlag = False
        #     var.set('already start')

    def clear():
        global pointlist, startFlag, stepFlag, loadnextFlag, listorder, Inputdata, count, inputnextcount, outputpointlist, outputedgelist
        canvas.delete("all")
        startFlag = False
        stepFlag = False
        loadnextFlag = False
        pointlist = []
        listorder = []
        Inputdata = []
        count = 0
        inputnextcount = 0
        outputpointlist = []
        outputedgelist = []

    def makeorder(list, stack):
        if len(list) == 1:
            stack.append(list)
            return stack
        Sl = []
        Sr = []
        median = np.mean(list, axis=0)
        if list:
            for point in list:
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
            stack = makeorder(Sl, stack)
            stack = makeorder(Sr, stack)
            stack.append(Sl+Sr)
        return stack

    def step():
        global stepFlag
        global listorder
        global count
        if stepFlag == False:
            listorder = []
            stepFlag = True
            listorder = makeorder(pointlist, [])
            print(listorder)
        else:
            if count == len(listorder):
                print("the end")
                clear()
                count = 0
            if count > 0:
                printpoint(listorder[count-1], 'blue')
            printpoint(listorder[count], 'red')
            count+=1

    def load():
        global loadnextFlag
        global Inputdata
        loadnextFlag = True
        def RepresentsInt(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        window.filename = filedialog.askopenfilename(initialdir="./", parent=window, title='Choose a file')

        file = open(window.filename, encoding="UTF-8")
        lines = []
        Inputdata = []
        if file:
            for line in file:
                lines.append(line.rstrip('\n'))

            for i in range(len(lines)):
                if RepresentsInt(lines[i]):
                    point = []
                    if int(lines[i]) == 0:
                        break
                    for j in range(i+1,i+int(lines[i])+1):
                        point.append(list(map(int, lines[j].split())))
                    Inputdata.append(point)
                else:
                    if lines[i] == "":
                        pass
                    elif lines[i][0] == "P":
                        newline = list(lines[i].split())
                        tmppoint = [float(newline[1]), float(newline[2])]
                        outputpointlist.append(tmppoint)
                        printpoint([tmppoint], 'black')
                    elif lines[i][0] == "E":
                        newline = list(lines[i].split())
                        tmpedge = [[float(newline[1]), float(newline[2])], [float(newline[3]), float(newline[4])]]
                        outputedgelist.append(tmpedge)
                        printedges([tmpedge], 'red')
            for thelist in Inputdata:
                print(thelist)
                print("----------------")

            file.close()


    def save():
        global outputedgelist, outputpointlist
        outputfile = open('outputfile.txt', 'w')
        outputstr = ""
        for p in outputpointlist:
            outputstr += "P "+str(p[0])+" "+str(p[1])+"\n"
        for e in outputedgelist:
            outputstr += "E " + str(e[0][0]) + " " + str(e[0][1]) + " "+ str(e[1][0]) + " " + str(e[1][1]) +"\n"
        outputfile.write(outputstr)
        outputfile.close()


    def inputnext(data):
        global pointlist
        global inputnextcount
        global loadnextFlag
        global currentPointlist
        if loadnextFlag == True:
            if inputnextcount == len(data):
                print("the end")
                clear()
                inputnextcount =0
                loadnextFlag = False
                currentPointlist.configure(text="Current PointList")
            elif inputnextcount >= 0:
                pointlist = data[inputnextcount]
                sumstr = ""
                for point in pointlist:
                    sumstr += "[%d, %d] " % (point[0], point[1])
                currentPointlist.configure(text=sumstr)

                inputnextcount += 1


    canvas = tk.Canvas(window, bg='#D0D0D0', height=600, width=600)
    canvas.bind('<Button 1>', left_click)
    canvas.grid(row=0, column=0, rowspan=5)


    startBtn = tk.Button(window, text='start', width=15, height=2, command=start)
    startBtn.grid(row=0,column=1)
    StepBtn = tk.Button(window, text='step', width=15, height=2, command=step)
    StepBtn.grid(row=1,column=1)
    cleanBtn = tk.Button(window, text='clear', width=15, height=2, command=clear)
    cleanBtn.grid(row=2,column=1)
    InputBtn = tk.Button(window, text='load', width=15, height=2, command=load)
    InputBtn.grid(row=3,column=1)
    OutputBtn = tk.Button(window, text='save', width=15, height=2, command=save)
    OutputBtn.grid(row=4, column=1)
    NextBtn = tk.Button(window, text='next', width=15, height=2, command=lambda: inputnext(Inputdata))
    NextBtn.grid(row=3, column=2)

    currentPointlist = tk.Label(window, text="Current PointList")
    currentPointlist.grid(row=5, column=0)


    window.mainloop()