import numpy as np

class Filter:

  #ponto
  p = np.array([[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]])

  #borda
  sob_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

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

  test = np.array([[0, 1, 1,   2,   2,   2, 1, 1, 0],
                   [1, 2, 4,   5,   5,   5, 4, 2, 1],
                   [1, 4, 5,   3,   0,   3, 5, 4, 1],
                   [2, 5, 3, -12, -24, -12, 3, 5, 2],
                   [2, 5, 0, -24, -40, -24, 0, 5, 2],
                   [2, 5, 3, -12, -24, -12, 3, 5, 2],
                   [1, 4, 5,   3,   0,   3, 5, 4, 1],
                   [1, 2, 4,   5,   5,   5, 4, 2, 1],
                   [0, 1, 1,   2,   2,   2, 1, 1, 0]])

  @staticmethod
  def gen_gaussiano(shape, dp=1.0):
    delta = shape//2
    return np.fromfunction(lambda i, j: ((np.e**(-(((i-delta)**2)+((j-delta)**2))/(2*(dp**2))))/((2*np.pi)*(dp**2))), (shape, shape), dtype= float)

  @staticmethod
  def gen_laplaceano_gaussiana(shape, dp=1.0):
    delta = shape//2

    return np.fromfunction(lambda i, j: -((1 - (((i-delta)**2 + (j-delta)**2)/(2*dp**2)))*np.e**(-((i-delta)**2+(j-delta)**2)/(2*dp**2)))/(np.pi*dp**4), (shape, shape), dtype = float)

  @staticmethod
  def gen_media_filter(shape):
    return np.ones((shape, shape), dtype= float)/(shape**2)

  @staticmethod
  def gen_sobel_filter(shape, axis=0):
    sobel = np.fromfunction(lambda i, j: -((shape//2)-j), (shape, shape), dtype= int)
    sobel[shape//2, :] = sobel[shape//2, :] * 2

    if axis == 0:
      return sobel
    else:
      return sobel.transpose()

  @staticmethod
  def gaussiano(img, size=3, dp=0.8):
    return Filter.apply(img, Filter.gen_gaussiano(size, dp))

  @staticmethod
  def laplaceano_gaussiana(img, size=3, dp=0.8):
    return Filter.apply(img, Filter.gen_laplaceano_gaussiana(size, dp))

  @staticmethod
  def media_filter(img, size=3):
    return Filter.apply(img, Filter.gen_media_filter(size))

  @staticmethod
  def sobel_filter(img, size=3):
    sobel = Filter.gen_sobel_filter(size)
    return Filter.apply(img, sobel) + Filter.apply(img, sobel.transpose())

  @staticmethod
  def apply(img, filtro):
    img = np.array(img)
    filtro_alt, filtro_larg = filtro.shape

    if len(img.shape) > 2:
      img = img.transpose(2,1,0)
      img_result = np.zeros_like(img)

      img_pad = []
      for k in img:
        img_pad.append(np.pad(k, (filtro_alt//2, filtro_larg//2), mode= 'constant', constant_values= 0))
    else:
      img_result = [np.zeros_like(img)]
      img_pad = [np.pad(img, (filtro_alt//2, filtro_larg//2), mode= 'constant', constant_values= 0)]
      img = [img]

    for k in range(len(img)):
      img_result[k] = np.clip(np.einsum('ijkl->ij', np.lib.stride_tricks.sliding_window_view(img_pad[k], filtro.shape)*filtro), 0, 255)

    if type(img)!=type([]) and len(img.shape) > 2:
      return img_result.transpose(2,1,0)
    else:
      return img_result[0]

  # imagem deve estar em tom de cinza
  @staticmethod
  def limiarizacao(img, erro= 0.01):
    img = np.array(img)
    img_result = np.zeros_like(img)

    media = img.mean()
    media_ = media*2

    while (media_ - media) > erro:
      media_ = media

      g1 = (img > media)
      g2 = np.logical_not(g1)

      media1 = np.mean(img[g1])
      media2 = np.mean(img[g2])

      media = (media1 + media2) / 2

    #print(media)

    particao = img > media

    img_result[particao] = 1

    return img_result

  @staticmethod
  def limiarizacao_otsu(img):
    thresholding_max = -1
    var_cinza_max = -1

    img = np.array(img)
    img_result = np.zeros_like(img)

    pos, hist = unique, counts = np.unique(img, return_counts=True)
    #print(hist)
    prob = hist/len(img)

    peso = lambda prob, r: np.sum(prob * r)
    media = lambda index, prob, r, peso: np.sum((index * prob) * r) / peso
    media_t = lambda index, prob, r: np.sum((index * prob) * r)
    var = lambda peso0, peso1, media0, media1, media_t: peso0 * ((media0 - media_t)**2) + peso1 * ((media1 - media_t)**2)

    p0 = 1
    p1 = 1
    m0 = 0
    m1 = 0
    index = np.fromfunction(lambda i: i, hist.shape)
    #print(index.shape, hist.shape)
    r = np.ones(len(hist))
    for k in range(1, len(hist)):
      p0 = peso(prob[0:k], r[0:k])
      p1 = peso(prob[k:len(hist)], r[k: len(hist)])
      m0 = media(index[0:k], prob[0:k], r[0:k], p0)
      m1 = media(index[k:len(hist)], prob[k:len(hist)], r[k: len(hist)], p1)
      m_t = media_t(index, prob, r)

      var_t = var(p0, p1, m0, m1, m_t)

      if var_t > var_cinza_max:
        var_cinza_max = var_t
        thresholding_max = k-1

    particao = img >= thresholding_max

    img_result[particao] = 1

    return img_result

  @staticmethod
  def limiarizacao_adapt(img, method= limiarizacao_otsu, qtd_partes= 2):
    img = np.array(img)
    img_result = np.zeros_like(img)

    x_dim = img.shape[0] // qtd_partes
    x_dim_ultimos = x_dim + (img.shape[0] % qtd_partes)

    y_dim = img.shape[1] // qtd_partes
    y_dim_ultimos = y_dim + (img.shape[1] % qtd_partes)

    for i in range(qtd_partes):
      for j in range(qtd_partes):
        desloc_x = x_dim if i < qtd_partes else x_dim_ultimos
        desloc_y = y_dim if j < qtd_partes else x_dim_ultimos

        img_result[i*desloc_x : (i+1)*desloc_x, j*desloc_y : (j+1)*y_dim] = method(img[i*desloc_x : (i+1)*desloc_x, j*desloc_y : (j+1)*desloc_y])

    return img_result

  @staticmethod
  def erosao(img, el= [[1,1,1],[1,1,1],[1,1,1]]):
    el = np.array(el)
    img = np.array(img)
    img_pad = np.pad(img, (el.shape[0]//2, el.shape[1]//2), mode= 'constant', constant_values= 0)
    img_result = np.zeros_like(img)

    parte = np.lib.stride_tricks.sliding_window_view(img_pad, el.shape)

    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        p = parte[i,j]
        if (p == el).all():
          img_result[i, j] = 1

    return img_result

  @staticmethod
  def dilatacao(img, el= [[1,1,1],[1,1,1],[1,1,1]]):
    el = np.array(el)
    img = np.array(img)
    img_pad = np.pad(img, (el.shape[0]//2, el.shape[1]//2), mode= 'constant', constant_values= 1)
    img_result = np.zeros_like(img)

    parte = np.lib.stride_tricks.sliding_window_view(img_pad, el.shape)

    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        p = parte[i,j]
        if (p == el).any():
          img_result[i, j] = 1

    return img_result

  @staticmethod
  def abertura(img, el= [[1,1,1],[1,1,1],[1,1,1]]):
    return Filter.dilatacao(Filter.erosao(img, el), el)

  @staticmethod
  def fechamento(img, el= [[1,1,1],[1,1,1],[1,1,1]]):
    return Filter.erosao(Filter.dilatacao(img, el), el)
