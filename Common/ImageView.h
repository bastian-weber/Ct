#ifndef HB_RASTERMAPRENDERER
#define HB_RASTERMAPRENDERER

//Qt
#include <QtWidgets/QtWidgets>
#include <QtGui/QtGui>

//OpenCv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//STL libraries
#include <vector>
#include <iostream>
#include <cmath>
#include <set>
#include <functional>

namespace hb {

	///A class for displaying images.
	/**
	 * The \c ImageView can display an image and allows the user can zoom and pan.
	 * Apart from that it supports adding and manipulation of points (e.g. reference points),
	 * overlay painting (e.g. a mask), overlay of a mask as well as overlay and manipulation
	 * of a polyline (e.g. a border).
	 */
	class ImageView : public QWidget {
		Q_OBJECT
	public:
		ImageView(QWidget *parent = 0);
		QSize sizeHint() const;

		void setShowInterfaceOutline(bool value);
		void setInterfaceBackgroundColor(QColor const& color);
		void setRightClickForHundredPercentView(bool value);
		bool getRightClickForHundredPercentView();
		void setUsePanZooming(bool value);
		bool getUsesPanZooming();

		void rotateLeft();
		void rotateRight();
		void setRotation(double degrees);
		void centerViewportOn(QPointF point);
		void setPreventMagnificationInDefaultZoom(bool value);

		void setImage(const QImage& image);
		void setImage(QImage&& image);
		void setImage(const cv::Mat& image);
		void setImageWithPrecomputedPreview(const cv::Mat& image, const cv::Mat& downscaledImage);
		void resetImage();
		bool getImageAssigned() const;
		QPointF mapToImageCoordinates(QPointF pointInWidgetCoordinates) const;

		double getCurrentPreviewScalingFactor() const;
		void setUseHighQualityDownscaling(bool value);
		bool getUseHighQualityDownscaling();
		void setUseSmoothTransform(bool value);
		bool getUseSmoothTransform() const;
		void setEnablePostResizeSharpening(bool value);
		bool getEnablePostResizeSharpening();
		void setPostResizeSharpeningStrength(double value);
		double getPostResizeSharpeningStrength();
		void setPostResizeSharpeningRadius(double value);
		double getPostResizeSharpeningRadius();
		void setPostResizeSharpening(bool enable, double strength, double radius);

		void setPointEditing(bool enablePointAdding, bool enablePointManipulation);
		void setRenderPoints(bool value);
		const std::vector<QPointF>& getPoints() const;
		void setPoints(const std::vector<QPointF>& points);
		void setPoints(std::vector<QPointF>&& points);
		void addPoint(const QPointF& point);
		void deleteOutsidePoints();

		void setPaintingActive(bool value);
		void setVisualizeBrushSize(bool value);
		const QBitmap& getMask() const;
		void setOverlayMask(const QBitmap& mask);
		void setOverlayMask(QBitmap&& mask);
		void setRenderOverlayMask(bool value);

		void setRenderRectangle(bool value);
		void setRectangle(QRectF rectangle);

		void setRenderPolyline(bool value);
		void setPolyline(std::vector<QPointF> border);
		//if this is activated it will disable point adding
		void setPolylineEditingActive(bool value);
		const std::vector<QPointF>& getPolyline() const;
		void setPolylineColor(QColor color);

		template <typename T>
		void setExternalPostPaintFunction(T* object, void(T::*function)(QPainter&));
		void setExternalPostPaintFunction(std::function<void(QPainter&)> const& function);
		void removeExternalPostPaintFunction();
	public slots:
		void zoomInKey();
		void zoomOutKey();
		void zoomToHundredPercent(QPointF center);
		void resetZoom();
		void resetMask();
		void setBrushRadius(int value);
		void deletePoint(int index);
		void resetPoints();
		void invertPolylineColor();
	protected:
		void showEvent(QShowEvent * e);
		void mousePressEvent(QMouseEvent* e);
		void mouseMoveEvent(QMouseEvent* e);
		void mouseReleaseEvent(QMouseEvent* e);
		void mouseDoubleClickEvent(QMouseEvent* e);
		void wheelEvent(QWheelEvent* e);
		void resizeEvent(QResizeEvent* e);
		void enterEvent(QEvent* e);
		void leaveEvent(QEvent* e);
		void paintEvent(QPaintEvent* e);
		void keyPressEvent(QKeyEvent * e);
		bool eventFilter(QObject *object, QEvent *e);
	private:
		double getEffectiveImageWidth() const;
		double getEffectiveImageHeight() const;
		double getWindowScalingFactor() const;
		QTransform getTransform() const;
		QTransform getTransformDownsampledImage() const;
		QTransform getTransformScaleRotateOnly() const;
		QTransform getTransformScaleOnly() const;
		QTransform getTransformRotateOnly() const;
		void zoomBy(double delta, QPointF const& center);
		void enforcePanConstraints();
		void updateResizedImage();

		static double distance(const QPointF& point1, const QPointF& point2);
		struct IndexWithDistance {
			IndexWithDistance(int index, double distance) : index(index), distance(distance) { };
			int index;
			double distance;
		};
		IndexWithDistance closestGrabbablePoint(QPointF const& mousePosition) const;
		IndexWithDistance closestGrabbablePolylinePoint(QPointF const& mousePosition) const;
		double smallestDistanceToPolyline(QPointF const& mousePosition) const;
		double smallestDistanceToPolylineSelection(QPointF const& mousePosition) const;
		static double distanceOfPointToLineSegment(QPointF const& lineStart, QPointF const& lineEnd, QPointF const& point);

		static void sharpen(cv::Mat& image, double strength, double radius);

		static void shallowCopyMatToImage(const cv::Mat& mat, QImage& destImage);
		static void deepCopyMatToImage(const cv::Mat& mat, QImage& destImage);
		static void shallowCopyImageToMat(const QImage& image, cv::Mat& destMat);
		static void deepCopyImageToMat(const QImage& image, cv::Mat& destMat);
		static void matToImage(const cv::Mat& mat, QImage& destImage, bool deepCopy);
		static void imageToMat(const QImage& image, cv::Mat& destMat, bool deepCopy);

		//related to general interface settings
		bool interfaceOutline;
		QColor backgroundColor;
		bool rightClickForHundredPercentView;
		bool usePanZooming;
		//the users transformations (panning, zooming)
		double zoomExponent;
		const double zoomBasis;
		bool preventMagnificationInDefaultZoom;
		bool hundredPercentZoomMode;
		QPointF panOffset;
		double viewRotation;
		//related to general click and drag events
		bool dragging;
		QPointF lastMousePosition;
		int screenId;
		bool moved;
		//related to pan-zooming
		bool panZooming;
		QPointF initialMousePosition;
		QPointF infinitePanLastInitialMousePosition;
		double panZoomingInitialZoomExponent;
		QPointF panZoomingInitialPanOffset;
		//related to setting points and rendering them
		std::vector<QPointF> points;
		bool pointEditingActive;
		bool pointManipulationActive;
		bool renderPoints;
		//related to editing of points
		double pointGrabTolerance;
		bool pointGrabbed;
		int grabbedPointIndex;
		bool showPointDeletionWarning;
		//related to displaying the image
		QImage image;
		cv::Mat mat;
		bool isMat;
		QImage downsampledImage;
		cv::Mat downsampledMat;
		bool imageAssigned;
		bool useHighQualityDownscaling;
		bool useSmoothTransform;
		bool enablePostResizeSharpening;
		double postResizeSharpeningStrength;
		double postResizeSharpeningRadius;
		//related to mask painting
		QBitmap mask;
		bool paintingActive;
		bool maskInitialized;
		bool painting;
		double brushRadius;
		QPointF brushPosition;
		bool visualizeBrushSize;
		//related to overlaying a mask
		QBitmap overlayMask;
		bool overlayMaskSet;
		bool renderOverlayMask;
		//related to painting rectangle
		QRectF rectangle;
		bool renderRectangle;
		//related to displaying a polyline
		std::vector<QPointF> polyline;
		bool polylineAssigned;
		bool renderPolyline;
		//related to editing the polyline
		bool polylineManipulationActive;
		bool polylinePointGrabbed;
		std::set<int> polylineSelectedPoints;
		double polylinePointGrabTolerance;
		bool polylineSelected;
		int polylineLastAddedPoint;
		QRectF selectionRectangle;
		std::set<int> selectionRectanglePoints;
		bool spanningSelectionRectangle;
		QColor polylineColor;
		//related to external post paint function
		std::function<void(QPainter&)> externalPostPaint;
		bool externalPostPaintFunctionAssigned;
	signals:
		///Emitted when a point is moved, emitted live during interaction (not just on mouse release).
		void pointModified();
		///Emitted when the user deletes a point, \p index specifies the index of the point that was deleted.
		void userDeletedPoint(int index);
		///Emitted when the user clicks somewhere where the click is not handled internally.
		/**
		 * A click is when the mouse is not moved inbetween a mouse press event and a mouse release
		 * event. Some of the internal features of the \c image view use clicks for certain actions.
		 * However, not every click might trigger internal procedures. If certain interaction features
		 * are disabled this is most likely. In this case this signal is emitted. It can be connected
		 * to a slot that then handles the occurence.
		 */
		void pixelClicked(QPoint pixel);
		///Emitted when the polyline was modified, not emitted live during interaction but on mouse release.
		void polylineModified();
	};


//=============================================================== IMPLEMENTATION OF TEMPLATE FUNCTIONS ===============================================================\\

	///Registers a member function \p function of an \p object that will be called at the end of the \c paintEvent method.
	/**
	* This method can be used to register the member function of an object as post-paint function.
	* The corresponding function will be called at the end of the \c paintEvent method.
	* To that function the current widget is passed as a \c QPainter object which enables custom
	* drawing on top of the widget, e.g. to display additional information.
	*/
	//template function, thus implemented in header
	template <typename T>
	void ImageView::setExternalPostPaintFunction(T* object, void(T::*function)(QPainter&)) {
		this->externalPostPaint = std::bind(function, object, std::placeholders::_1);
		this->externalPostPaintFunctionAssigned = true;
	}

}

#endif