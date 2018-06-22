import cv2 as cv
import numpy as np

# Crear una instancia del filtro kalman

class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)

    kf.measurementMatrix = np.array([[1, 0, 0, 0], 
				     [0, 1, 0, 0]], np.float32)

    kf.transitionMatrix = np.array([[1, 0, 1, 0],
				    [0, 1, 0, 1], 
				    [0, 0, 1, 0],
				    [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):

        ''' Esta función estima la posición del objeto'''

        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])

        self.kf.correct(measured)

        predicted = self.kf.predict()

        return predicted

#Realiza el procesamiento de imágenes requerido para que la bola se coordine en el video

class ProcessImage:

    def DetectObject(self):

        vid = cv.VideoCapture(1)

        if(vid.isOpened() == False):

            print('Cannot open input video')

            return

        width = int(vid.get(3))

        height = int(vid.get(4))

        # Crear objeto de filtro de Kalman

        kfObj = KalmanFilter()

        predictedCoords = np.zeros((2, 1), np.float32)

        while(vid.isOpened()):

            rc, frame = vid.read()

            if(rc == True):

                [ballX, ballY] = self.DetectBall(frame)

                predictedCoords = kfObj.Estimate(ballX, ballY)
                
                
                
                # Dibujar coordenadas actuales de la segmentación

                cv.circle(frame, (ballX, ballY), 20, [0,0,255], 2, 8)

                cv.line(frame,(ballX, ballY + 20), (ballX + 50, ballY + 20), [0,0,255], 2,8)

                cv.putText(frame, "Actual", (ballX + 50, ballY + 20), cv.FONT_HERSHEY_SIMPLEX,0.5, [0,0,255])


                # Dibujar la salida de prediccion de Filtro de Kalman

                cv.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0,255,0], 2, 8)

                cv.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [0, 255, 0], 2, 8)

                cv.putText(frame, "Prediccion", (predictedCoords[0] + 50, predictedCoords[1] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0])

                cv.imshow('Input', frame)

                if (cv.waitKey(5) & 0xFF == ord('q')):

                    break
            else:

                break

        vid.release()

        cv.destroyAllWindows()

    # Segmenta la bola verde en un marco dado

    def DetectBall(self, frame):


        #Convertimos de RGB -> HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        h,w,m=frame.shape
        
    # Establecer el umbral para filtrar solo el color verde y filtrarlo

        bajos = np.array([0, 0, 255], dtype=np.uint8)
        altos = np.array([100, 100, 255], dtype=np.uint8)

        greenMask = cv.inRange(hsv, bajos, altos)

        #Divide la imagen en cuatro cuadrantes
        cv.line(frame,(0,h/2),(w,h/2),(150,200,0),2)        
        cv.line(frame,(w/2,0),(w/2,h),(150,200,0),2) 

        # Dilatar

        kernel = np.ones((5, 5), np.uint8)

        greenMaskDilated = cv.dilate(greenMask, kernel)

        cv.imshow('Thresholded', greenMaskDilated)

        # Encuentra bola detectada ya que es el objeto verde más grande en el marco

        [nLabels, labels, stats, centroids] = cv.connectedComponentsWithStats(greenMaskDilated)

        # El primer contorno más grande es siempre el borde de la imagen, elimínelo

        stats = np.delete(stats, (0), axis = 0)

        i, j = np.unravel_index(stats.argmax(), stats.shape)

        # Esta es nuestra bola de coordenadas que necesita ser rastreado

        x = stats[i, 0] + (stats[i, 2]/2)

        y = stats[i, 1] + (stats[i, 3]/2)
        #print "Coordenadas: ", (x-w/2),(-(y-h/2))
        #x1=x-w/2
        #y1=-(y-h/2)
        return [x,y]

#Función principal

def main():

    processImg = ProcessImage()

    processImg.DetectObject()

if __name__ == "__main__":

    main()

print('Program Completed!')
