import numpy as np

class YOLOKMeans:
    
    def __init__(self, num_clusters, filename):
        self.num_clusters = num_clusters
        self.filename = filename

    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        k = self.num_clusters

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, (1, n))
        cluster_area = np.reshape(cluster_area, (n, k))
        
        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

        inter_area = np.multiply(min_h_matrix, min_w_matrix)
        result = inter_area / (box_area + cluster_area - inter_area)

        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number, ))

        np.random.seed()

        clusters = boxes[np.random.choice(box_number, k, replace=False)]

        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)

            if (last_nearest == current_nearest).all():
                break

            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def resulttotext(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]

        for i in range(row):
            x_y = "%d,%d" % (data[i][0], data[i][1])
            
            if(i != 0) x_y = ", " + x_y

            f.write(x_y)
        f.close()

    def texttoboxes(self):
        f = open(self.filename, 'r')
        dataset = []

        for line in f:
            infos = line.split(" ")
            length = len(infos)

            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])

                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])

                dataset.append([width, height])
        result = np.array(dataset)
        f.close()
        return result

    def texttoclusters(self): 
        all_boxes = self.texttoboxes()
        result = self.kmeans(all_boxes, k=self.num_clusters)
        result = result[np.lexsort(result.T[0, None])]
        self.resulttotext(result)

        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
    kmeans = YOLOKMeans(num_clusters=9, filename=" ")
    kmeans.texttoclusters()
