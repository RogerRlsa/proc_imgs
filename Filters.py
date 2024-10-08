import numpy as np

class Filter:
  
    #borda
    sob_x = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
    
    sob_y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]])
    
    laplace = np.array([[ 0,  0, -1,  0,  0],
                        [ 0, -1, -2, -1,  0],
                        [-1, -2, 16, -2, -1],
                        [ 0, -1, -2, -1,  0],
                        [ 0,  0, -1,  0,  0]])

    #suavização
    gauss = np.array([[1,  4,  7,  4, 1],
                      [4, 16, 26, 16, 4],
                      [7, 26, 41, 26, 7],
                      [4, 16, 26, 16, 4],
                      [1,  4,  7,  4, 1]])/273

    media = np.array([[1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]])/25
    
    @staticmethod
    def gaussiano(img):
        return Filter.apply(img, Filter.gauss)

    @staticmethod
    def laplaciano(img):
        return Filter.apply(img, Filter.laplace)

    @staticmethod
    def media_filter(img):
        return Filter.apply(img, Filter.media)

    @staticmethod
    def sobel_filter(img):
        return Filter.apply(img, Filter.sob_y) + Filter.apply(img, Filter.sob_x)
    
    @staticmethod
    def apply(img, filtro):
        img = np.array(img)
        filtro_alt, filtro_larg = filtro.shape

        if len(img.shape) >= 2:
            img = img.transpose(2,1,0)
            img_result = np.zeros_like(img)

            img_pad = []
            for k in img:
                img_pad.append(np.pad(k, (filtro_alt//2, filtro_larg//2), mode= 'constant', constant_values= 0.0))
        else:
            img_result = [np.zeros_like(img)]
            img_pad = [np.pad(img, (filtro_alt//2, filtro_larg//2), mode= 'constant', constant_values= 0.0)]
            img = [img]

        for k in range(len(img)):
            img_result[k] = np.clip(np.einsum('ijkl->ij', np.lib.stride_tricks.sliding_window_view(img_pad[k], filtro.shape)*filtro), 0, 255)
            # v1
            #for i in range(img[0].shape[0]):
            #    for j in range(img[0].shape[1]):
            #        img_parte = img_pad[k][i: i + filtro_alt, j:j + filtro_larg]
            #        img_result[k][i, j]= np.clip(np.sum(img_parte * filtro), 0, 255)

        if type(img)!=type([]) and len(img.shape) >= 2:
            return img_result.transpose(2,1,0)
        else:
            return img_result[0]
    