import streamlit as st
from fastai.vision import *
import PIL.Image

html_temp = """
    <div style="background-color:ghostwhite;padding:10px;margin-bottom: 25px">
    <h2 style="color:black;text-align:center;">Moroccan Pastries Classifier</h2>
    <p style="color:black;text-align:center;" >This is a <b>Streamlit</b> app used to classify <b>10 common Moroccan pastries</b>.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

proj_path = 'Moroccan Pastries images/'
p_path = Path(proj_path)

@st.cache(max_entries=10, ttl=3600)

def data():
    data = ImageDataBunch.from_folder(p_path, train=".", valid_pct=0.3,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
    return data

learner = load_learner('')

option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose your own image':
  uploaded_file = st.file_uploader("Choose an image...", type="jpg")
  if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    import torchvision.transforms as tfms
    def pil2fast(img):  
      return Image(tfms.ToTensor()(img))
    pred_class,pred_idx,outputs = learner.predict(pil2fast(img))
    st.image(img)
    st.write(pred_class)
if option == 'Choose a test image':
  test_images = os.listdir('Sample images')
  test_image = st.selectbox('Please select a test image:', test_images)
  file_path = 'Sample images/' + test_image
  img = PIL.Image.open(file_path)
  import torchvision.transforms as tfms
  def pil2fast(img):  
    return Image(tfms.ToTensor()(img))
  pred_class,pred_idx,outputs = learner.predict(pil2fast(img))
  st.image(img)
  st.write(pred_class)
