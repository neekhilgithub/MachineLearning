import streamlit as st
import geopandas as gpd
from samgeo.text_sam import LangSAM
from samgeo import tms_to_geotiff
from shapely.geometry import Point

def download_google_data():
    st.title("Google Data Downloader")

    # Specify a Google basemap to use
    basemap_choice = st.selectbox("Select a Google basemap", ["ROADMAP", "TERRAIN", "SATELLITE", "HYBRID"])

    # Prompt the user to enter bounding box information
    st.subheader("Enter the bounding box information")
    bbox_input = st.text_input("Bounding Box (xmin, ymin, xmax, ymax)", "-51.253043,-22.17615,-51.2498,-22.1739")
    bbox = [float(coord.strip()) for coord in bbox_input.split(',')]

    # Prompt the user to enter the file name for the image
    output_filename = st.text_input("Enter the file name for the image (without extension)", "output")

    # State variable to keep track of download success
    if 'download_success' not in st.session_state:
        st.session_state.download_success = False

    # Convert TMS to GeoTIFF image
    if st.button("Download GeoTIFF"):
        st.write("Please wait...")
        try:
            tms_to_geotiff(output=output_filename + ".tif", bbox=bbox, zoom=19, source=basemap_choice, overwrite=True)
            st.session_state.download_success = True
            st.write("Download successful!")
        except Exception as e:
            st.error(f"Error: {e}")

    # Display the "Show GeoTIFF" button only if download was successful
    if st.session_state.download_success:
        if st.button("Show GeoTIFF"):
            st.image(output_filename + ".tif")

def use_sam_model():
    st.title('Image Processing with LangSAM')

    # Get input image path from user
    image_path = st.text_input("Enter the path to the input image: ")

    # Get text prompt from user
    text_prompt = st.text_area("Enter the text prompt: ")
    
    box_threshold = st.slider("Enter the box_threshold:", min_value=0.0, max_value=1.0, step=0.01)
    
    text_threshold = st.slider("Enter the text_threshold:", min_value=0.0, max_value=1.0, step=0.01)

    # Define output filename for raster
    output_filename = st.text_input("Enter the output filename for raster (without extension): ", value="output").strip()

    # Define output filename for vector
    output_vector_filename = st.text_input("Enter the output filename for vector (without extension): ", value="vector").strip()

    if st.button("Process"):
        try:
            # Initialize LangSAM model
            sam = LangSAM()

            # Predict using LangSAM model
            sam.predict(image_path, text_prompt, box_threshold, text_threshold)

            # Show annotations
            sam.show_anns(
                cmap='Greys_r',
                add_boxes=False,
                alpha=1,
                title='Automatic Segmentation',
                blend=False,
                output=output_filename + '.tif',  # Using the defined output filename
            )

            # Convert raster to vector
            sam.raster_to_vector(output_filename + ".tif", output_vector_filename + ".shp")

            # Convert polygons to points
            output_point_filename = output_vector_filename + "_points.shp"
            gdf = gpd.read_file(output_vector_filename + ".shp")
            gdf['geometry'] = gdf['geometry'].centroid
            gdf.to_file(output_point_filename)

            st.success("Raster to Vector conversion and Polygon to Point conversion completed successfully.")

            # Count the number of points
            st.write("Total number of points:", len(gdf))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def main():
    st.title("Satellite Data and SAM Model")
    choice = st.radio("Choose an option:", ("Download Satellite Data", "Use SAM Model"))

    if choice == "Download Satellite Data":
        download_google_data()
    elif choice == "Use SAM Model":
        use_sam_model()

if __name__ == "__main__":
    main()