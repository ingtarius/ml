from sklearn import svm

x=[]
y=[]
for i in range(0,100):
    mylist=[]
    mylist.append(i)
    mylist.append(i)
    x.append(mylist)
    if(i%2)==0:
        y.append(0)
    else:
        y.append(1)

clf = svm.SVC()
clf.fit(x, y)

print(x)
print(y)
x_pred = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11],[12,12],[13,13],[14,14],[15,15],[16,16]]
y_pred=clf.predict(x_pred)
print(y_pred)
