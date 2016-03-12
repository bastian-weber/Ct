#include "ImageView.h"

namespace hb {

	//========================================================================= Public =========================================================================\\

	ImageView::ImageView(QWidget *parent)
		: QWidget(parent),
		interfaceOutline(true),
		useHighQualityDownscaling(true),
		rightClickForHundredPercentView(true),
		usePanZooming(true),
		imageAssigned(false),
		isMat(false),
		zoomBasis(1.5),
		zoomExponent(0),
		preventMagnificationInDefaultZoom(false),
		hundredPercentZoomMode(false),
		panOffset(0, 0),
		viewRotation(0),
		dragging(false),
		pointEditingActive(false),
		pointManipulationActive(false),
		renderPoints(false),
		moved(false),
		panZooming(false),
		paintingActive(false),
		maskInitialized(false),
		brushRadius(5),
		brushPosition(0, 0),
		painting(false),
		visualizeBrushSize(false),
		pointGrabTolerance(10),
		pointGrabbed(false),
		showPointDeletionWarning(false),
		overlayMaskSet(false),
		renderOverlayMask(false),
		renderRectangle(false),
		polylineAssigned(false),
		renderPolyline(false),
		useSmoothTransform(true),
		enablePostResizeSharpening(false),
		postResizeSharpeningStrength(0.5),
		postResizeSharpeningRadius(1),
		polylineManipulationActive(false),
		polylinePointGrabbed(false),
		polylineSelected(false),
		polylinePointGrabTolerance(10),
		polylineLastAddedPoint(0),
		spanningSelectionRectangle(false),
		polylineColor(60, 60, 60),
		externalPostPaintFunctionAssigned(false) {
		setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
		setFocusPolicy(Qt::FocusPolicy::StrongFocus);
		setMouseTracking(true);
		QPalette palette = qApp->palette();
		this->backgroundColor = palette.base().color();
	}

	QSize ImageView::sizeHint() const {
		return QSize(360, 360);
	}

	///Defines wether an outline should be drawn around the widget (indicating also if it has the focus or not)
	void ImageView::setShowInterfaceOutline(bool value) {
		this->interfaceOutline = value;
	}

	///Sets the background colour or the widget.
	void ImageView::setInterfaceBackgroundColor(QColor const& color) {
		this->backgroundColor = color;
	}

	///If set to true, a right click zooms the image to 100% magnification.
	void ImageView::setRightClickForHundredPercentView(bool value) {
		this->rightClickForHundredPercentView = value;
	}

	///Returns \c true if the right click for 100% view feature is enabled.
	bool ImageView::getRightClickForHundredPercentView() {
		return this->rightClickForHundredPercentView;
	}

	///If set to true, panning while holding the middle mouse button will change the zoom.
	void ImageView::setUsePanZooming(bool value) {
		this->usePanZooming = true;
	}

	///Returns true if pan-zooming is enabled.
	bool ImageView::getUsesPanZooming() {
		return this->usePanZooming;
	}

	///Rotates the viewport 90° in anticlockwise direction.
	void ImageView::rotateLeft() {
		this->viewRotation -= 90;
		if (this->viewRotation < 0) this->viewRotation += 360;
		this->enforcePanConstraints();
		this->updateResizedImage();
		if (this->isVisible()) this->update();
	}

	///Rotates the viewport 90° in clockwise direction.
	void ImageView::rotateRight() {
		this->viewRotation += 90;
		if (this->viewRotation >= 360) this->viewRotation -= 360;
		this->enforcePanConstraints();
		this->updateResizedImage();
		if (this->isVisible()) this->update();
	}

	///Sets the rotation of the view to \p degrees degrees.
	void ImageView::setRotation(double degrees) {
		this->viewRotation = degrees;
		if (this->viewRotation >= 360) this->viewRotation = this->viewRotation - (360 * std::floor(degrees / 360.0));
		if (this->viewRotation < 0) this->viewRotation = this->viewRotation + (360 * std::ceil(std::abs(degrees / 360.0)));
		this->enforcePanConstraints();
		this->updateResizedImage();
		if (this->isVisible()) this->update();
	}

	///Moves the viewport to the point \p point.
	void ImageView::centerViewportOn(QPointF point) {
		QPointF transformedPoint = this->getTransform().map(point);
		this->panOffset += (QPointF((double)this->width() / 2.0, (double)this->height() / 2.0) - transformedPoint) / (pow(this->zoomBasis, this->zoomExponent)*this->getWindowScalingFactor());
		this->enforcePanConstraints();
		this->update();
	}

	///If set to true, images won't be enlarged at the default magnification (fit view).
	void ImageView::setPreventMagnificationInDefaultZoom(bool value) {
		this->preventMagnificationInDefaultZoom = value;
		this->update();
	}

	///Makes the \c ImageView display the image \p image, shallow copy assignment.
	void ImageView::setImage(const QImage& image) {
		QSize oldSize = this->image.size();
		this->image = image;
		//free the mat
		this->isMat = false;
		this->mat = cv::Mat();
		if (this->image.size() != oldSize) {
			this->resetMask();
			this->hundredPercentZoomMode = false;
		}

		this->imageAssigned = true;
		this->updateResizedImage();
		this->enforcePanConstraints();
		this->update();
	}

	///Makes the \c ImageView display the image \p image, move assignment.
	void ImageView::setImage(QImage&& image) {
		QSize oldSize = this->image.size();
		this->image = std::move(image);
		//free the mat
		this->isMat = false;
		this->mat = cv::Mat();
		if (this->image.size() != oldSize) {
			this->resetMask();
			this->hundredPercentZoomMode = false;
		}

		this->imageAssigned = true;
		this->updateResizedImage();
		this->enforcePanConstraints();
		this->update();
	}

	///Makes the \c ImageView display the image \p image, shallow copy assignment.
	void ImageView::setImage(const cv::Mat& image) {
		if (image.type() == CV_8UC4 || image.type() == CV_8UC3 || image.type() == CV_8UC1) {
			QSize oldSize = this->image.size();
			this->mat = image;
			ImageView::shallowCopyMatToImage(this->mat, this->image);
			if (this->image.size() != oldSize) {
				this->resetMask();
				this->hundredPercentZoomMode = false;
			}
			this->isMat = true;

			this->imageAssigned = true;
			this->updateResizedImage();
			this->enforcePanConstraints();
			this->update();
		} else {
			std::cerr << "Image View: cannot assign image because of unsupported type " << image.type() << "." << std::endl;
		}
	}

	///Identical to setImage(const cv::Mat& image) but with a precalculated resized version.
	/**
	 * This funciton can be used to speed up the display process. In \p downscaledImage an
	 * image can be passed that has the dimensions needed for the current zoom level. Thus
	 * the \c ImageView will not have to compute this itself but will just use the provided one.
	 * This offers the possibility to calculate this externally where it can potentially be
	 * done faster (e.g. on the GPU) and save some computation time.
	 */
	void ImageView::setImageWithPrecomputedPreview(const cv::Mat& image, const cv::Mat& downscaledImage) {
		if ((image.type() == CV_8UC4 || image.type() == CV_8UC3 || image.type() == CV_8UC1) && (downscaledImage.type() == CV_8UC4 || downscaledImage.type() == CV_8UC3 || downscaledImage.type() == CV_8UC1)) {
			QSize oldSize = this->image.size();
			this->mat = image;
			ImageView::shallowCopyMatToImage(this->mat, this->image);
			if (this->image.size() != oldSize) {
				this->resetMask();
				this->hundredPercentZoomMode = false;
			}
			this->downsampledMat = downscaledImage;
			ImageView::shallowCopyMatToImage(this->downsampledMat, this->downsampledImage);
			this->isMat = true;

			this->imageAssigned = true;
			this->update();
		} else {
			std::cerr << "Image View: cannot assign image and downsampled preview because at least one is of unsupported type (" << image.type() << " and " << downscaledImage.type() << ")." << std::endl;
		}
	}

	///Removes the image.
	void ImageView::resetImage() {
		this->image = QImage();
		this->imageAssigned = false;
		this->update();
	}

	///Returns \c true if an image is assigned, false otherwise.
	bool ImageView::getImageAssigned() const {
		return this->imageAssigned;
	}

	///Maps a point in widget coordinates to image coordinates of the currently assigned image
	QPointF ImageView::mapToImageCoordinates(QPointF pointInWidgetCoordinates) const {
		if (this->imageAssigned) {
			QPointF result = this->getTransform().inverted().map(pointInWidgetCoordinates);
			if (result.x() >= 0 && result.y() >= 0 && result.x() < double(this->image.width()) && result.y() < double(this->image.height())) {
				return result;
			}
			return QPointF();
		}
		return QPointF();
	}

	///Returns the magnification factor at which the image is dispayed, 1 means the image is at a 100% view and one image pixel corresponds to one pixel of the display.
	double ImageView::getCurrentPreviewScalingFactor() const {
		if (this->imageAssigned) {
			return std::pow(this->zoomBasis, this->zoomExponent) * this->getWindowScalingFactor();
		} else {
			return -1;
		}
	}

	///Specifies if the image will be resampled with a high quality algorithm when it's displayed with a magnificaiton smaller than 1.
	void ImageView::setUseHighQualityDownscaling(bool value) {
		this->useHighQualityDownscaling = value;
		this->updateResizedImage();
		this->update();
	}

	///Returns \c true if high quality downscaling is enabled, \c false otherwise.
	bool ImageView::getUseHighQualityDownscaling() {
		return this->useHighQualityDownscaling;
	}

	///Specifies if the sampling will be done bilinear or nearest neighbour when the iamge is displayed at a magnification greater than 1.
	void ImageView::setUseSmoothTransform(bool value) {
		this->useSmoothTransform = value;
		this->update();
	}

	///Returns \c true if bilinear sampling is enabled, \c false otherwise.
	bool ImageView::getUseSmoothTransform() const {
		return this->useSmoothTransform;
	}

	///If enabled the image will be unsharped masked after it has been downsampled to the current zoom level.
	/**
	* When resizing images to lower resolutions their sharpness impression might suffer.
	* By enabling this feature images will be sharpened after they have resampled to a
	* smaller size. The strenght and radius of this sharpening filter can be set via
	* \c ImageView::setPostResizeSharpeningStrength(double value) and
	* \c ImageView::setPostResizeSharpeningRadius(double value). If the zoom level is
	* at a level at which the image does not have to be downsampled, no sharpening
	* filter will be applied.
	*/
	void ImageView::setEnablePostResizeSharpening(bool value) {
		this->enablePostResizeSharpening = value;
		this->updateResizedImage();
		this->update();
	}

	///Returns \c true if the post resize sharpening is enabled, \c false otherwise.
	bool ImageView::getEnablePostResizeSharpening() {
		return this->enablePostResizeSharpening;
	}

	///Sets the strength value of the post-resize unsharp masking filter to \p value.
	void ImageView::setPostResizeSharpeningStrength(double value) {
		this->postResizeSharpeningStrength = value;
		this->updateResizedImage();
		this->update();
	}

	///Returns the strength value of the post-resize unsharp masking filter.
	double ImageView::getPostResizeSharpeningStrength() {
		return this->postResizeSharpeningStrength;
	}

	///Sets the radius value of the post-resize unsharp masking filter to \p value.
	void ImageView::setPostResizeSharpeningRadius(double value) {
		this->postResizeSharpeningRadius = value;
		this->updateResizedImage();
		this->update();
	}

	///Returns the radius value of the post-resize unsharp masking filter.
	double ImageView::getPostResizeSharpeningRadius() {
		return this->postResizeSharpeningRadius;
	}

	///Sets all parameters for the post resize sharpening at once.
	/**
	* The advantage of using this function instead of setting the three
	* parameters separately is that the imageView will only have to update once,
	* resulting in better performance.
	*/
	void ImageView::setPostResizeSharpening(bool enable, double strength, double radius) {
		this->enablePostResizeSharpening = enable;
		this->postResizeSharpeningStrength = strength;
		this->postResizeSharpeningRadius = radius;
		this->updateResizedImage();
		this->update();
	}

	///Enables or disables the ability to set new points and the ability to move already set ones; if adding points is enabled, manipulation of the polyline will be disabled.
	void ImageView::setPointEditing(bool enablePointAdding, bool enablePointManipulation) {
		this->pointEditingActive = enablePointAdding;
		this->pointManipulationActive = enablePointManipulation;
		if (enablePointAdding || enablePointManipulation)this->renderPoints = true;
		if (enablePointAdding && this->polylineManipulationActive) {
			std::cout << "Point adding was enabled, thus polyline manipulation will be disabled." << std::endl;
			this->polylineManipulationActive = false;
		}
	}

	///Specpfies whether points are rendered or not.
	void ImageView::setRenderPoints(bool value) {
		this->renderPoints = value;
		this->update();
	}

	///Returns the currently set points.
	const std::vector<QPointF>& ImageView::getPoints() const {
		return this->points;
	}

	///Sets the points to \p points.
	void ImageView::setPoints(const std::vector<QPointF>& points) {
		this->points = points;
		this->update();
	}

	///Sets the points to \p points.
	void ImageView::setPoints(std::vector<QPointF>&& points) {
		this->points = std::move(points);
		this->update();
	}

	///Adds the point \p point.
	void ImageView::addPoint(const QPointF& point) {
		this->points.push_back(point);
		this->update();
	}

	///Deletes all points which are outside the image, might happen when new image is assigned.
	void ImageView::deleteOutsidePoints() {
		for (std::vector<QPointF>::iterator point = this->points.begin(); point != this->points.end();) {
			if (point->x() < 0 || point->x() >= this->image.width() || point->y() < 0 || point->y() >= this->image.height()) {
				emit(userDeletedPoint(point - this->points.begin()));
				point = this->points.erase(point);
			} else {
				++point;
			}
		}
		this->update();
	}

	///Specifies whether it's possible to do overlay painting or not.
	/**
	 * Painting allows the user, for example, to mask certain areas.
	 */
	void ImageView::setPaintingActive(bool value) {
		if (value == true && !this->maskInitialized && this->imageAssigned) {
			this->mask = QBitmap(this->image.size());
			this->mask.fill(Qt::color0);
			this->maskInitialized = true;
		}
		if (value == false || this->imageAssigned) {
			this->paintingActive = value;
		}
	}

	///Specifies whether brush size visualization is enabled or not.
	/**
	 * If this is enabled a brush of the currently assigned brush size will be displayed in the
	 * center of the \c ImageView. Use \c setBrushRadius(int value) to set the brush size. This can
	 * be used to provide feedback to the user as to how large the brush currently is e.g. when
	 * its size is being changed.
	 */
	void ImageView::setVisualizeBrushSize(bool value) {
		this->visualizeBrushSize = value;
		this->update();
	}

	///Returns the mask that has been painted by the user.
	const QBitmap& ImageView::getMask() const {
		return this->mask;
	}

	///Sets a mask that will be displayed as a half-transparent overlay.
	/**
	 * This mask is not related to the panting the user does; instead it is an additional layer.
	 * This overload does a shallow copy assignment.
	 */
	void ImageView::setOverlayMask(const QBitmap& mask) {
		this->overlayMask = mask;
		this->overlayMaskSet = true;
		this->update();
	}

	///Sets a mask that will be displayed as a half-transparent overlay.
	/**
	* This mask is not related to the panting the user does; instead it is an additional layer.
	* This overload does a move assignment.
	*/
	void ImageView::setOverlayMask(QBitmap&& mask) {
		this->overlayMask = std::move(mask);
		this->overlayMaskSet = true;
		this->update();
	}

	///Specifies whether the assigned overlay mask is rendered or not.
	void ImageView::setRenderOverlayMask(bool value) {
		this->renderOverlayMask = value;
		this->update();
	}

	void ImageView::setRenderRectangle(bool value) {
		this->renderRectangle = value;
		this->update();
	}

	void ImageView::setRectangle(QRectF rectangle) {
		this->rectangle = rectangle;
		this->update();
	}

	///Specifies whether the assigned polyline is rendered or not.
	void ImageView::setRenderPolyline(bool value) {
		this->renderPolyline = value;
		this->update();
	}

	///Assigns a polyline that can be overlayed.
	void ImageView::setPolyline(std::vector<QPointF> border) {
		this->polyline = border;
		this->polylineAssigned = true;
		this->update();
	}

	///Enables or disables the ability to edit the polyline, will disable the ability to add points.
	void ImageView::setPolylineEditingActive(bool value) {
		this->polylineManipulationActive = value;
		if (value && this->pointEditingActive) {
			std::cout << "Polyline editing was enabled, thus point adding will be disabled." << std::endl;
			this->pointEditingActive = false;
		}
	}

	///Returns the polyline as it currently is.
	const std::vector<QPointF>& ImageView::getPolyline() const {
		return this->polyline;
	}

	///Sets the colour that the polyline is rendered in.
	void ImageView::setPolylineColor(QColor color) {
		this->polylineColor = color;
		this->update();
	}

	///Registers any callable target so it will be called at the end of the \c paintEvent method.
	/**
	* This method can be used to register any \c std::function as post-paint function.
	* Also function pointers or lambdas can be passed. They then will be implicitly converted.
	* The corresponding function will be called at the end of the \c paintEvent method.
	* To that function the current widget is passed as a \c QPainter object which enables custom
	* drawing on top of the widget, e.g. to display additional information.
	*/
	void ImageView::setExternalPostPaintFunction(std::function<void(QPainter&)> const& function) {
		this->externalPostPaint = function;
		this->externalPostPaintFunctionAssigned = true;
	}

	//Removes any assigned post-paint function, which then is no longer invoked.
	void ImageView::removeExternalPostPaintFunction() {
		this->externalPostPaintFunctionAssigned = false;
	}

	//========================================================================= Public Slots =========================================================================\\

	///Zooms the viewport in one step.
	void ImageView::zoomInKey() {
		QPointF center = QPointF(double(this->width()) / 2.0, double(this->height()) / 2.0);
		if (this->underMouse()) center = this->mapFromGlobal(QCursor::pos());
		this->zoomBy(1, center);
	}

	///Zooms the viewport out one step.
	void ImageView::zoomOutKey() {
		QPointF center = QPointF(double(this->width()) / 2.0, double(this->height()) / 2.0);
		if (this->underMouse()) center = this->mapFromGlobal(QCursor::pos());
		this->zoomBy(-1, center);
	}

	///Resets the mask the user is painting, does not affect the overlay mask.
	void ImageView::resetMask() {
		if (this->maskInitialized) {
			this->mask = QBitmap(this->image.size());
			this->mask.fill(Qt::color0);
			this->update();
		}
	}

	///Sets the radius of the brush to \p value.
	void ImageView::setBrushRadius(int value) {
		this->brushRadius = value;
		if (this->visualizeBrushSize) {
			this->update();
		}
	}

	///Displays the image at 100% magnification; the point \p center (in widget screen coordinates) will be centered.
	void ImageView::zoomToHundredPercent(QPointF center) {
		if (this->imageAssigned) {
			QPointF mousePositionCoordinateBefore = this->getTransform().inverted().map(center);
			double desiredZoomFactor = 1 / this->getWindowScalingFactor();
			this->zoomExponent = log(desiredZoomFactor) / log(this->zoomBasis);
			QPointF mousePositionCoordinateAfter = this->getTransform().inverted().map(center);
			//remove the rotation from the delta
			QPointF mouseDelta = this->getTransformRotateOnly().map(mousePositionCoordinateAfter - mousePositionCoordinateBefore);
			this->panOffset += mouseDelta;
			this->hundredPercentZoomMode = true;
			this->enforcePanConstraints();
			this->updateResizedImage();
			this->update();
		}
	}

	void ImageView::resetZoom() {
		this->zoomExponent = 0;
		this->hundredPercentZoomMode = false;
		this->enforcePanConstraints();
		this->updateResizedImage();
		this->update();
	}

	///Deletes the point at index \p index.
	void ImageView::deletePoint(int index) {
		if (index >= 0 && index < this->points.size()) {
			this->points.erase(this->points.begin() + index);
			this->update();
		}
	}

	///Removes all the set points.
	void ImageView::resetPoints() {
		this->points.clear();
		this->update();
	}

	///Inverts the colour that the assigned polyline is rendered in.
	void ImageView::invertPolylineColor() {
		this->polylineColor = QColor(255 - this->polylineColor.red(), 255 - this->polylineColor.green(), 255 - this->polylineColor.blue());
		this->update();
	}

	//========================================================================= Protected =========================================================================\\

	void ImageView::showEvent(QShowEvent * e) {
		this->enforcePanConstraints();
	}

	void ImageView::mousePressEvent(QMouseEvent *e) {
		this->lastMousePosition = e->pos();
		this->screenId = qApp->desktop()->screenNumber(QCursor::pos());
		this->initialMousePosition = e->pos();
		this->infinitePanLastInitialMousePosition = this->initialMousePosition;

		if (e->modifiers() & Qt::AltModifier && this->polylineManipulationActive && this->polylineAssigned) {
			//span a selection rectangle
			this->polylineSelected = true;
			this->selectionRectangle = QRectF(e->pos(), e->pos());
			if (!(e->modifiers() & Qt::ControlModifier)) {
				this->polylineSelectedPoints.clear();
			}
			this->spanningSelectionRectangle = true;
		} else {
			//check for close points to grab
			IndexWithDistance closestPoint = this->closestGrabbablePoint(e->pos());
			IndexWithDistance closestPolylinePoint = this->closestGrabbablePolylinePoint(e->pos());
			double polylineSelectionDistance = this->smallestDistanceToPolylineSelection(e->pos());
			if (closestPoint.index >= 0 && (closestPolylinePoint.index <= 0 || closestPoint.distance < closestPolylinePoint.distance || !this->polylineSelected) && (closestPoint.distance < polylineSelectionDistance || this->polylineSelectedPoints.size() == 0 || !this->polylineSelected) && this->pointManipulationActive) {
				//grab a point
				this->grabbedPointIndex = closestPoint.index;
				qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
				this->pointGrabbed = true;
			} else if ((closestPolylinePoint.index >= 0 || (polylineSelectionDistance <= this->polylinePointGrabTolerance && this->polylineSelectedPoints.size() > 0)) && this->polylineManipulationActive && this->polylineSelected) {
				//polyline editing
				if (((polylineSelectionDistance <= closestPolylinePoint.distance || this->polylineSelectedPoints.find(closestPolylinePoint.index) != this->polylineSelectedPoints.end()) && this->polylineSelectedPoints.size() > 0) || closestPolylinePoint.index < 0 && !(e->modifiers() & Qt::ControlModifier) && !(e->modifiers() & Qt::ShiftModifier)) {
					//start moving the selection
					this->polylinePointGrabbed = true;
				} else {
					if (closestPolylinePoint.index >= 0) {
						if (e->modifiers() & Qt::ShiftModifier) {
							if (!e->modifiers() & Qt::ControlModifier) this->polylineSelectedPoints.clear();
							//add all points inbetween the current point and the last point
							int largerIndex = std::max(closestPolylinePoint.index, this->polylineLastAddedPoint);
							int smallerIndex = std::min(closestPolylinePoint.index, this->polylineLastAddedPoint);
							for (int index = smallerIndex; index <= largerIndex; ++index) {
								this->polylineSelectedPoints.insert(index);
							}
						} else {
							if (e->modifiers() & Qt::ControlModifier) {
								//add point to selected points or remove it
								std::set<int>::iterator point = this->polylineSelectedPoints.find(closestPolylinePoint.index);
								if (point == this->polylineSelectedPoints.end()) {
									this->polylineSelectedPoints.insert(closestPolylinePoint.index);
								} else {
									this->polylineSelectedPoints.erase(point);
								}
								this->polylineLastAddedPoint = closestPolylinePoint.index;
							} else {
								this->polylineSelectedPoints.clear();
								//grab polyline point
								this->polylineSelectedPoints.insert(closestPolylinePoint.index);
								this->polylineLastAddedPoint = closestPolylinePoint.index;
								this->polylinePointGrabbed = true;
							}
						}
					}
				}
			} else if ((!this->paintingActive && e->button() != Qt::MiddleButton) || (this->paintingActive && e->button() == Qt::MiddleButton)) {
				//dragging
				this->dragging = true;		
			} else if (e->button() == Qt::MiddleButton) {
				//pan-zooming
				this->panZooming = true;
				this->panZoomingInitialPanOffset = this->panOffset;
				this->panZoomingInitialZoomExponent = this->zoomExponent;
				qApp->setOverrideCursor(QCursor(Qt::SizeVerCursor));
			} else if (this->imageAssigned) {
				//painting
				this->painting = true;

				//paint a circle
				QPainter canvas(&this->mask);
				canvas.setPen(Qt::NoPen);
				if (e->button() == Qt::LeftButton) {
					canvas.setBrush(QBrush(Qt::color1));
				} else {
					canvas.setBrush(QBrush(Qt::color0));
				}
				QTransform transform = this->getTransform().inverted();
				canvas.drawEllipse(transform.map(QPointF(e->pos())), this->brushRadius, this->brushRadius);
				this->update();
			}
		}

		this->moved = false;
	}

	void ImageView::mouseMoveEvent(QMouseEvent *e) {
		this->moved = true;
		bool dontUpdateLastMousePosition = false;

		if (this->dragging || this->pointGrabbed || this->polylinePointGrabbed) {
			QPointF deltaRotated = this->getTransformScaleRotateOnly().inverted().map((e->pos() - this->lastMousePosition));
			QPointF deltaScaled = this->getTransformScaleOnly().inverted().map((e->pos() - this->lastMousePosition));
			if (this->dragging) {
				//dragging
				qApp->setOverrideCursor(QCursor(Qt::ClosedHandCursor));
				this->panOffset += deltaScaled;
				this->enforcePanConstraints();
				//for infinite panning
				QPoint globalPos = QCursor::pos();
				QRect screen = QApplication::desktop()->screen(this->screenId)->geometry();
				QPoint newPos;
				if (globalPos.y() >= screen.bottom()) {
					newPos = QPoint(globalPos.x(), screen.top() + 1);
				} else if (globalPos.y() <= screen.top()) {
					newPos = QPoint(globalPos.x(), screen.bottom() - 1);
				} else if (globalPos.x() >= screen.right()) {
					newPos = QPoint(screen.left() + 1, globalPos.y());
				} else if (globalPos.x() <= screen.left()) {
					newPos = QPoint(screen.right() - 1, globalPos.y());
				}
				if (newPos != QPoint()) {
					this->lastMousePosition = mapFromGlobal(newPos);
					dontUpdateLastMousePosition = true;
					QCursor::setPos(newPos);
				}
			} else if (this->pointGrabbed) {
				//editing points
				this->points[this->grabbedPointIndex] += deltaRotated;
				if (e->pos().x() < 0 || e->pos().y() < 0 || e->pos().x() > this->width() || e->pos().y() > this->height() || this->points[this->grabbedPointIndex].x() < 0 || this->points[this->grabbedPointIndex].y() < 0 || this->points[this->grabbedPointIndex].x() >= this->image.width() || this->points[this->grabbedPointIndex].y() >= this->image.height()) {
					this->showPointDeletionWarning = true;
					qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
				} else {
					this->showPointDeletionWarning = false;
					qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
				}
				emit pointModified();
			} else {
				//editing polyline points
				for (int index : this->polylineSelectedPoints) {
					this->polyline[index] += deltaRotated;
					if (this->polyline[index].x() < 0)this->polyline[index].setX(0);
					if (this->polyline[index].x() > this->image.width())this->polyline[index].setX(this->image.width());
					if (this->polyline[index].y() < 0)this->polyline[index].setY(0);
					if (this->polyline[index].y() > this->image.height())this->polyline[index].setY(this->image.height());
				}
			}
			this->update();
		} else if (this->spanningSelectionRectangle) {
			this->selectionRectangle.setBottomLeft(e->pos());
			QTransform transform = this->getTransform();
			this->selectionRectanglePoints.clear();
			for (int point = 0; point < this->polyline.size(); ++point) {
				QPointF transformedPoint = transform.map(this->polyline[point]);
				if (this->selectionRectangle.contains(transformedPoint)) {
					this->selectionRectanglePoints.insert(point);
					this->polylineLastAddedPoint = point;
				}
			}
			this->update();
		}

		if (this->paintingActive) {
			this->brushPosition = e->pos();
			if (this->painting) {
				//draw a line from last mouse position to the current
				QPainter canvas(&this->mask);
				QPen pen;
				if (e->buttons() == Qt::LeftButton) {
					pen.setColor(Qt::color1);
				} else {
					pen.setColor(Qt::color0);
				}
				pen.setWidth(2 * this->brushRadius);
				pen.setCapStyle(Qt::RoundCap);
				canvas.setPen(pen);
				QTransform transform = this->getTransform().inverted();
				canvas.drawLine(transform.map(this->lastMousePosition), transform.map(e->pos()));
			}
			this->update();
		}

		if (this->panZooming) {
			this->zoomExponent = this->panZoomingInitialZoomExponent;
			this->panOffset = this->panZoomingInitialPanOffset;
			double delta = (this->infinitePanLastInitialMousePosition - e->pos()).y() * (-0.025);
			this->zoomBy(delta, this->initialMousePosition);
			//for infinite pan zooming
			QPoint globalPos = QCursor::pos();
			QRect screen = QApplication::desktop()->screen(this->screenId)->geometry();
			QPoint newPos;
			if (globalPos.y() >= screen.bottom()) {
				newPos = QPoint(globalPos.x(), screen.top() + 1);
			} else if (globalPos.y() <= screen.top()) {
				newPos = QPoint(globalPos.x(), screen.bottom() - 1);
			} else if (globalPos.x() >= screen.right()) {
				newPos = QPoint(screen.left() + 1, globalPos.y());
			} else if (globalPos.x() <= screen.left()) {
				newPos = QPoint(screen.right() - 1, globalPos.y());
			}
			if (newPos != QPoint()) {
				this->infinitePanLastInitialMousePosition = mapFromGlobal(newPos);
				this->panZoomingInitialPanOffset = this->panOffset;
				this->panZoomingInitialZoomExponent = this->zoomExponent;
				QCursor::setPos(newPos);
			}
			//doesn't work as expected
			//QCursor::setPos(mapToGlobal(this->lastMousePosition.toPoint()));
		}

		if (!this->dragging && !this->painting && !this->pointGrabbed && !this->spanningSelectionRectangle && !this->panZooming) {
			//check for close points to grab
			if (this->pointManipulationActive) {
				if (this->closestGrabbablePoint(e->pos()).index >= 0) {
					qApp->setOverrideCursor(QCursor(Qt::OpenHandCursor));
				} else {
					qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
				}
			}
		}

		if ((this->dragging || this->painting || this->pointGrabbed || this->polylinePointGrabbed) && !dontUpdateLastMousePosition) {
			this->lastMousePosition = e->pos();
		}
	}

	void ImageView::mouseReleaseEvent(QMouseEvent *e) {

		//clicking points
		if (this->pointEditingActive && this->imageAssigned && !this->moved) {
			//this was a click, add a point
			QTransform transform = this->getTransform();
			QPointF clickedPoint = e->pos();
			QPointF worldPoint = transform.inverted().map(clickedPoint);
			if (worldPoint.x() >= 0 && worldPoint.x() <= this->image.width() && worldPoint.y() >= 0 && worldPoint.y() <= this->image.height()) {
				this->points.push_back(worldPoint);
				std::cout << "Point added: " << worldPoint.x() << "  " << worldPoint.y() << std::endl;
				emit pointModified();
			}
		} else if (!this->pointEditingActive && !this->moved && this->imageAssigned) {
			if (this->polylineManipulationActive && e->button() != Qt::RightButton) {
				//this was a click, select or unselect polyline
				if (this->smallestDistanceToPolyline(e->pos()) <= this->polylinePointGrabTolerance) {
					//clicked close enough to a point, select line
					this->polylineSelected = true;
				} else {
					//clicked somewehere else, deselect it
					this->polylineSelected = false;
					this->polylineSelectedPoints.clear();
				}
			}

			if (e->button() == Qt::RightButton && this->rightClickForHundredPercentView) {
				//zoom to 100%
				if (this->hundredPercentZoomMode) {
					this->resetZoom();
				} else {
					this->zoomToHundredPercent(e->pos());
				}
			}

			//emit pixel click signal
			QTransform transform = this->getTransform();
			QPointF clickedPoint = e->pos();
			QPointF worldPoint = transform.inverted().map(clickedPoint);
			emit(pixelClicked(QPoint(std::floor(worldPoint.x()), std::floor(worldPoint.y()))));
		}

		if (this->pointGrabbed) {
			if (e->pos().x() < 0 || e->pos().y() < 0 || e->pos().x() > this->width() || e->pos().y() > this->height() || this->points[this->grabbedPointIndex].x() < 0 || this->points[this->grabbedPointIndex].y() < 0 || this->points[this->grabbedPointIndex].x() >= this->image.width() || this->points[this->grabbedPointIndex].y() >= this->image.height()) {
				this->deletePoint(this->grabbedPointIndex);
				emit(userDeletedPoint(this->grabbedPointIndex));
				this->showPointDeletionWarning = false;
				qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			} else {
				qApp->setOverrideCursor(QCursor(Qt::OpenHandCursor));
			}
			this->pointGrabbed = false;
			emit pointModified();
		}

		if (this->polylinePointGrabbed) {
			this->polylinePointGrabbed = false;
			if (this->moved) emit polylineModified();
		}

		if (this->spanningSelectionRectangle) {
			this->spanningSelectionRectangle = false;
			this->polylineSelectedPoints.insert(this->selectionRectanglePoints.begin(), this->selectionRectanglePoints.end());
			this->selectionRectanglePoints.clear();
		}

		if (this->dragging) {
			if (this->paintingActive) {
				qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
			} else {
				qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			}
			this->dragging = false;
		}
		this->painting = false;

		if (this->panZooming) {
			qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			this->panZooming = false;
		}

		this->update();
	}

	void ImageView::mouseDoubleClickEvent(QMouseEvent* e) {
		e->ignore();
	}

	void ImageView::wheelEvent(QWheelEvent* e) {
		if (!this->panZooming) {
			double divisor = 1;
			if (e->modifiers() & Qt::ControlModifier) {
				divisor = 600;
			} else if (!e->modifiers()) {
				divisor = 120;
			} else {
				e->ignore();
				return;
			}
			this->zoomBy(e->delta() / divisor, e->pos());
		}
		e->accept();
	}

	void ImageView::resizeEvent(QResizeEvent* e) {
		//maintain 100% view if in 100% view
		if (this->hundredPercentZoomMode) {
			QPointF center(this->width() / 2.0, this->height() / 2.0);
			this->zoomToHundredPercent(center);
		}
		this->updateResizedImage();
	}

	void ImageView::enterEvent(QEvent* e) {
		if (this->paintingActive) {
			qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
		}
	}

	void ImageView::leaveEvent(QEvent* e) {
		qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
		if (this->paintingActive) {
			this->update();
		}
	}

	void ImageView::paintEvent(QPaintEvent* e) {
		QPainter canvas(this);
		canvas.setRenderHint(QPainter::Antialiasing, true);
		canvas.setRenderHint(QPainter::SmoothPixmapTransform, this->useSmoothTransform);
		QSize canvasSize = this->size();
		QTransform transform = this->getTransform();
		QPalette palette = qApp->palette();
		canvas.fillRect(0, 0, this->width(), this->height(), this->backgroundColor);

		//drawing of the image
		if (this->imageAssigned) {
			if (std::pow(this->zoomBasis, this->zoomExponent) * this->getWindowScalingFactor() >= 1 || !this->useHighQualityDownscaling) {
				canvas.setTransform(transform);
				canvas.drawImage(QPoint(0, 0), this->image);
			} else {
				canvas.setTransform(this->getTransformDownsampledImage());
				canvas.drawImage(QPoint(0, 0), this->downsampledImage);
			}
		}

		//drawing of the overlay mask
		if (this->overlayMaskSet && this->renderOverlayMask) {
			canvas.setTransform(transform);
			QImage image = this->overlayMask.toImage();
			image.setColor(Qt::color0, QColor(Qt::white).rgb());
			image.setColor(Qt::color1, Qt::transparent);
			canvas.setOpacity(0.9);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, false);
			canvas.drawImage(QPoint(0, 0), image);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, true);
			canvas.setOpacity(1);
		}

		//drawing of bounds (rectangle) overlay
		if (this->imageAssigned && this->renderRectangle) {
			QPixmap rect = QPixmap(canvasSize);
			rect.fill(Qt::transparent);
			QRectF imageArea(QPointF(0, 0), this->image.size());
			imageArea = transform.mapRect(imageArea);
			QPainter p(&rect);
			p.setRenderHint(QPainter::Antialiasing, true);
			p.setPen(Qt::NoPen);
			p.setBrush(QColor(0, 0, 0, 100));
			p.drawRect(imageArea);
			p.setBrush(QBrush(Qt::transparent));
			p.setCompositionMode(QPainter::CompositionMode_SourceOut);
			QRectF eraseRect = transform.mapRect(this->rectangle);
			p.drawRect(eraseRect);
			canvas.resetTransform();
			canvas.drawPixmap(0, 0, rect);
		}

		//drawing of the mask that is currently painted
		if (this->paintingActive) {
			canvas.setTransform(transform);
			QImage image = this->mask.toImage();
			image.setColor(Qt::color0, Qt::transparent);
			image.setColor(Qt::color1, QColor(Qt::red).rgb());
			canvas.setOpacity(0.5);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, false);
			canvas.drawImage(QPoint(0, 0), image);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, true);
			canvas.setOpacity(1);
		}

		//drawing of the polyline if assigned
		if (this->polylineAssigned && this->renderPolyline && this->polyline.size() > 0) {
			canvas.setRenderHint(QPainter::Antialiasing, true);
			canvas.setTransform(transform);
			QPen linePen = QPen(this->polylineColor);
			linePen.setJoinStyle(Qt::MiterJoin);
			if (!this->polylineSelected || !this->polylineManipulationActive) linePen.setWidth(3);
			QBrush brush = QBrush(this->polylineColor);
			linePen.setCosmetic(true);
			canvas.setPen(linePen);
			if (this->polyline.size() > 1) {
				canvas.drawPolyline(this->polyline.data(), this->polyline.size());
				if (this->polylineManipulationActive && this->polylineSelected) {
					canvas.resetTransform();
					const int squareSize = 4;
					const int squareOffset = squareSize / 2;
					for (int point = 0; point < this->polyline.size(); ++point) {
						if (this->selectionRectanglePoints.find(point) != this->selectionRectanglePoints.end() || this->polylineSelectedPoints.find(point) != this->polylineSelectedPoints.end()) {
							canvas.setBrush(brush);
						} else {
							canvas.setBrush(Qt::NoBrush);
						}
						QPointF transformedPoint = transform.map(this->polyline[point]);
						canvas.drawRect(transformedPoint.x() - squareOffset, transformedPoint.y() - squareOffset, squareSize, squareSize);
					}
				}
			} else {
				canvas.drawPoint(this->polyline[0]);
			}
		}

		//draw selection rectangle when selecting
		if (this->spanningSelectionRectangle) {
			canvas.resetTransform();
			canvas.setPen(QPen(Qt::darkGray, 1, Qt::DashDotLine));
			canvas.setBrush(Qt::NoBrush);
			canvas.drawRect(this->selectionRectangle);
		}

		//drawing of the points
		if (this->renderPoints) {
			canvas.resetTransform();
			QPen pen(Qt::black, 2);
			pen.setCosmetic(true);
			QPen textPen(palette.buttonText().color());
			QBrush brush(Qt::white);
			canvas.setBrush(brush);
			QFont font;
			font.setPointSize(10);
			canvas.setFont(font);
			QColor base = palette.base().color();
			base.setAlpha(200);
			canvas.setBackground(base);
			canvas.setBackgroundMode(Qt::OpaqueMode);
			QPointF transformedPoint;
			for (int point = 0; point < this->points.size(); ++point) {
				transformedPoint = transform.map(this->points[point]);
				canvas.setPen(pen);
				canvas.drawEllipse(transformedPoint, 5, 5);
				canvas.setPen(textPen);
				canvas.drawText(transformedPoint + QPointF(7.0, 14.0), QString::number(point + 1));
			}
			if (this->pointEditingActive) {
				canvas.setPen(textPen);
				QString statusMessage = ((this->points.size() != 1) ? QString(tr("There are ")) : QString(tr("There is "))) + QString::number(this->points.size()) + ((this->points.size() != 1) ? QString(tr(" points set.")) : QString(tr(" point set.")));
				canvas.drawText(QPoint(20, this->height() - 15), statusMessage);
			}
		}

		//if painting active draw brush outline
		if (this->paintingActive && this->underMouse() && !this->dragging) {
			canvas.resetTransform();
			double scalingFactor = pow(this->zoomBasis, this->zoomExponent) * this->getWindowScalingFactor();
			canvas.setBrush(Qt::NoBrush);
			canvas.setPen(QPen(Qt::darkGray, 1));
			canvas.drawEllipse(this->brushPosition, this->brushRadius*scalingFactor, this->brushRadius*scalingFactor);
		}

		//visualization of the brush size (e.g. when changing it)
		if (this->visualizeBrushSize) {
			canvas.resetTransform();
			canvas.setPen(QPen(Qt::darkGray));
			canvas.setBrush(Qt::NoBrush);
			double scalingFactor = pow(this->zoomBasis, this->zoomExponent) * this->getWindowScalingFactor();
			canvas.drawEllipse(QPointF((double)this->width() / 2.0, (double)this->height() / 2.0), this->brushRadius*scalingFactor, this->brushRadius*scalingFactor);
		}

		//the point deletion warning
		if (this->showPointDeletionWarning) {
			canvas.resetTransform();
			QFont font;
			font.setPointSize(20);
			canvas.setFont(font);
			QColor base = palette.base().color();
			base.setAlpha(200);
			canvas.setBackground(base);
			canvas.setBackgroundMode(Qt::OpaqueMode);
			QPen textPen(palette.buttonText().color());
			canvas.setPen(textPen);
			canvas.drawText(QRect(0, 0, this->width(), this->height()), Qt::AlignCenter, QString(tr("Release to delete point")));
		}

		//add a contour
		if (this->interfaceOutline) {
			canvas.resetTransform();
			canvas.setRenderHint(QPainter::Antialiasing, 0);
			QColor strokeColour;
			if (this->hasFocus()) {
				strokeColour = palette.highlight().color();
			} else {
				strokeColour = palette.base().color();
				strokeColour.setRed(strokeColour.red() / 2);
				strokeColour.setGreen(strokeColour.green() / 2);
				strokeColour.setBlue(strokeColour.blue() / 2);
			}
			canvas.setPen(QPen(strokeColour, 1));
			canvas.setBrush(Qt::NoBrush);
			canvas.drawRect(0, 0, this->width() - 1, this->height() - 1);
		}

		//call external post paint function
		if (this->externalPostPaintFunctionAssigned) {
			canvas.resetTransform();
			this->externalPostPaint(canvas);
		}
	}

	void ImageView::keyPressEvent(QKeyEvent * e) {
		if ((this->isVisible() && (this->underMouse() || e->key() == Qt::Key_X) && this->imageAssigned) || e->key() == Qt::Key_S) {
			if (e->key() == Qt::Key_Plus && !this->panZooming) {
				this->zoomInKey();
			} else if (e->key() == Qt::Key_Minus && !this->panZooming) {
				this->zoomOutKey();
			} else if (e->key() == Qt::Key_S) {
				this->setUseSmoothTransform(!this->useSmoothTransform);
			} else if (e->key() == Qt::Key_X && this->polylineAssigned && this->renderPolyline) {
				this->invertPolylineColor();
			} else {
				e->ignore();
			}
		} else {
			e->ignore();
		}
	}

	bool ImageView::eventFilter(QObject *object, QEvent *e) {
		if (e->type() == QEvent::KeyPress) {
			QKeyEvent* keyEvent = (QKeyEvent*)e;
			if ((keyEvent->key() == Qt::Key_Plus || keyEvent->key() == Qt::Key_Minus) && this->isVisible() && this->underMouse() && this->imageAssigned) {
				this->keyPressEvent(keyEvent);
				return true;
			} else if (keyEvent->key() == Qt::Key_S) {
				this->keyPressEvent(keyEvent);
			} else if (keyEvent->key() == Qt::Key_X && this->isVisible() && this->imageAssigned && this->polylineAssigned && this->renderPolyline) {
				this->keyPressEvent(keyEvent);
				return true;
			}
		}
		return false;
	}

	//========================================================================= Private =========================================================================\\

	double ImageView::getEffectiveImageWidth() const {
		return std::abs(std::cos(this->viewRotation * M_PI / 180)) * (double)this->image.width() + std::abs(std::sin(this->viewRotation * M_PI / 180)) * (double)this->image.height();
	}

	double ImageView::getEffectiveImageHeight() const {
		return std::abs(std::cos(this->viewRotation * M_PI / 180)) * (double)this->image.height() + std::abs(std::sin(this->viewRotation * M_PI / 180)) * (double)this->image.width();
	}

	double ImageView::getWindowScalingFactor() const {
		if (this->imageAssigned && this->image.width() != 0 && this->image.height() != 0) {
			double imageWidth = this->getEffectiveImageWidth();
			double imageHeight = this->getEffectiveImageHeight();
			double scalingFactor = std::min((double)this->size().width() / imageWidth, (double)this->size().height() / imageHeight);
			if (this->preventMagnificationInDefaultZoom && scalingFactor > 1) {
				return 1;
			} else {
				return scalingFactor;
			}
		} else {
			return 1;
		}
	}

	QTransform ImageView::getTransform() const {
		//makes the map always fill the whole interface element
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		double centeringOffsetX = (double)this->image.width() / 2;
		double centeringOffsetY = (double)this->image.height() / 2;
		double transX = ((this->width() / factor) - this->image.width()) / 2;
		double transY = ((this->height() / factor) - this->image.height()) / 2;
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		transform.scale(factor, factor);
		//translation that ensures that the image is always centered in the view
		transform.translate(transX, transY);
		//move left upper corner back to 0, 0
		transform.translate(centeringOffsetX, centeringOffsetY);
		//apply users zoom
		transform.scale(zoomFactor, zoomFactor);
		//apple users pan
		transform.translate(this->panOffset.x(), this->panOffset.y());
		//rotate the view
		transform.rotate(this->viewRotation);
		//move image center to 0, 0
		transform.translate((-1)*centeringOffsetX, (-1)*centeringOffsetY);

		return transform;
	}

	QTransform ImageView::getTransformDownsampledImage() const {
		//makes the map always fill the whole interface element
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		/*Here we can do integer division for the centering offset because this function is only called
		when the image is displayed with negative magnificaiton. The Error of ca. 0.5 pixels can be
		accepted in this case because it will not be visible very much. Floating point numbers in
		contrast whould result in a slightly blurred image when the image is rotated and one ofset
		is integer while the other one is a fraction (because of the difference when moving the image to
		the origin and moving the image back would be < 1px due to the intermediate roation)*/
		double centeringOffsetX = this->downsampledImage.width() / 2;
		double centeringOffsetY = this->downsampledImage.height() / 2;
		double transX = ((this->width()) - this->downsampledImage.width()) / 2;
		double transY = ((this->height()) - this->downsampledImage.height()) / 2;
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		//transform.scale(factor, factor);
		//translation that ensures that the image is always centered in the view
		transform.translate(transX, transY);
		//move left upper corner back to 0, 0
		transform.translate(centeringOffsetX, centeringOffsetY);
		//apply users zoom
		//transform.scale(zoomFactor, zoomFactor);
		//apple users pan
		transform.translate(this->panOffset.x() * zoomFactor * factor, this->panOffset.y() * zoomFactor * factor);
		//rotate the view
		transform.rotate(this->viewRotation);
		//move image center to 0, 0
		transform.translate((-1)*centeringOffsetX, (-1)*centeringOffsetY);

		return transform;
	}

	QTransform ImageView::getTransformScaleRotateOnly() const {
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		transform.scale(factor, factor);
		//apply users zoom
		transform.scale(zoomFactor, zoomFactor);
		//rotate the view
		transform.rotate(this->viewRotation);
		return transform;
	}

	QTransform ImageView::getTransformScaleOnly() const {
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		transform.scale(factor, factor);
		//apply users zoom
		transform.scale(zoomFactor, zoomFactor);
		return transform;
	}

	QTransform ImageView::getTransformRotateOnly() const {
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//rotate the view
		transform.rotate(this->viewRotation);
		return transform;
	}

	void ImageView::zoomBy(double delta, QPointF const& center) {
		if (this->imageAssigned) {
			QPointF mousePositionCoordinateBefore = this->getTransform().inverted().map(center);
			this->zoomExponent += delta;
			if (this->zoomExponent < 0)this->zoomExponent = 0;
			QPointF mousePositionCoordinateAfter = this->getTransform().inverted().map(center);
			//remove the rotation from the delta
			QPointF mouseDelta = this->getTransformRotateOnly().map(mousePositionCoordinateAfter - mousePositionCoordinateBefore);
			this->panOffset += mouseDelta;
			this->hundredPercentZoomMode = false;
			this->enforcePanConstraints();
			this->updateResizedImage();
			this->update();
		}
	}

	void ImageView::enforcePanConstraints() {
		double imageWidth = this->getEffectiveImageWidth();
		double imageHeight = this->getEffectiveImageHeight();
		double factor = this->getWindowScalingFactor();
		double zoomFactor = pow(this->zoomBasis, this->zoomExponent);
		double maxXOffset = (-1)*(((this->width() / factor / zoomFactor) - imageWidth) / 2);
		double maxYOffset = (-1)*(((this->height() / factor / zoomFactor) - imageHeight) / 2);
		maxXOffset = std::max(0.0, maxXOffset);
		maxYOffset = std::max(0.0, maxYOffset);
		if (this->panOffset.x() > maxXOffset)this->panOffset.setX(maxXOffset);
		if (this->panOffset.x() < (-1) * maxXOffset)this->panOffset.setX((-1) * maxXOffset);
		if (this->panOffset.y() > maxYOffset)this->panOffset.setY(maxYOffset);
		if (this->panOffset.y() < (-1) * maxYOffset)this->panOffset.setY((-1) * maxYOffset);
	}

	void ImageView::updateResizedImage() {
		if (this->useHighQualityDownscaling && this->imageAssigned) {
			double scalingFactor = std::pow(this->zoomBasis, this->zoomExponent) * this->getWindowScalingFactor();
			if (scalingFactor < 1) {
				if (!this->isMat) {
					if (this->image.format() == QImage::Format_RGB888 || this->image.format() == QImage::Format_Indexed8 || this->image.format() == QImage::Format_ARGB32) {
						cv::Mat orig;
						ImageView::shallowCopyImageToMat(this->image, orig);
						cv::resize(orig, this->downsampledMat, cv::Size(), scalingFactor, scalingFactor, cv::INTER_AREA);
						if (this->enablePostResizeSharpening) {
							ImageView::sharpen(this->downsampledMat, this->postResizeSharpeningStrength, this->postResizeSharpeningRadius);
						}
						ImageView::deepCopyMatToImage(this->downsampledMat, this->downsampledImage);
					} else {
						//alternative
						this->downsampledImage = this->image.scaledToWidth(this->image.width() * scalingFactor, Qt::SmoothTransformation);
					}
				} else {
					cv::resize(this->mat, this->downsampledMat, cv::Size(), scalingFactor, scalingFactor, cv::INTER_AREA);
					if (this->enablePostResizeSharpening) {
						ImageView::sharpen(this->downsampledMat, this->postResizeSharpeningStrength, this->postResizeSharpeningRadius);
					}
					ImageView::shallowCopyMatToImage(this->downsampledMat, this->downsampledImage);
				}
			}
		}
	}

	double ImageView::distance(const QPointF& point1, const QPointF& point2) {
		return std::sqrt(std::pow(point2.x() - point1.x(), 2) + std::pow(point2.y() - point1.y(), 2));
	}

	ImageView::IndexWithDistance ImageView::closestGrabbablePoint(QPointF const& mousePosition) const {
		if (this->points.size() > 0) {
			QTransform transform = this->getTransform();
			double smallestDistance = this->distance(transform.map(this->points[0]), mousePosition);
			double index = 0;
			for (int point = 1; point < this->points.size(); ++point) {
				double tmpDistance = this->distance(transform.map(this->points[point]), mousePosition);
				if (tmpDistance < smallestDistance) {
					smallestDistance = tmpDistance;
					index = point;
				}
			}
			if (smallestDistance < this->pointGrabTolerance) {
				return IndexWithDistance(index, smallestDistance);
			}
		}
		return IndexWithDistance(-1, 0);
	}

	ImageView::IndexWithDistance ImageView::closestGrabbablePolylinePoint(QPointF const& mousePosition) const {
		if (this->polyline.size() > 0) {
			QTransform transform = this->getTransform();
			double smallestDistance = this->distance(transform.map(this->polyline[0]), mousePosition);
			double index = 0;
			for (int point = 1; point < this->polyline.size(); ++point) {
				double tmpDistance = this->distance(transform.map(this->polyline[point]), mousePosition);
				if (tmpDistance < smallestDistance) {
					smallestDistance = tmpDistance;
					index = point;
				}
			}
			if (smallestDistance < this->pointGrabTolerance) {
				return IndexWithDistance(index, smallestDistance);
			}
		}
		return IndexWithDistance(-1, 0);
	}

	double ImageView::smallestDistanceToPolyline(QPointF const& mousePosition) const {
		if (this->polyline.size() > 0) {
			QTransform transform = this->getTransform();
			double smallestDistance = this->distance(this->polyline[0], mousePosition);
			if (this->polyline.size() > 1) {
				for (int point = 0; point < this->polyline.size() - 1; ++point) {
					QPointF point1 = transform.map(this->polyline[point]);
					QPointF point2 = transform.map(this->polyline[point + 1]);
					double d = ImageView::distanceOfPointToLineSegment(point1, point2, mousePosition);
					if (d < smallestDistance) smallestDistance = d;
				}
			}
			return smallestDistance;
		}
		return 0;
	}

	double ImageView::smallestDistanceToPolylineSelection(QPointF const& mousePosition) const {
		if (this->polyline.size() > 0) {
			QTransform transform = this->getTransform();
			double smallestDistance = -1;
			for (int index : this->polylineSelectedPoints) {
				QPointF point1 = transform.map(this->polyline[index]);
				double d;
				if (this->polylineSelectedPoints.find(index + 1) != this->polylineSelectedPoints.end()) {
					//check distance to line segment	
					QPointF point2 = transform.map(this->polyline[index + 1]);
					d = ImageView::distanceOfPointToLineSegment(point1, point2, mousePosition);
				} else {
					//check distance to point
					d = this->distance(point1, mousePosition);
				}
				if (d < smallestDistance || smallestDistance == -1) smallestDistance = d;
			}
			return smallestDistance;
		}
		return 0;
	}

	double ImageView::distanceOfPointToLineSegment(QPointF const& lineStart, QPointF const& lineEnd, QPointF const& point) {
		QVector2D pointConnection(lineEnd - lineStart);
		QVector2D lineNormal(pointConnection);
		double tmp = lineNormal.x();
		lineNormal.setX(lineNormal.y());
		lineNormal.setY(tmp);
		if (lineNormal.x() != 0) {
			lineNormal.setX(lineNormal.x() * (-1));
		} else {
			lineNormal.setX(lineNormal.y() * (-1));
		}
		lineNormal.normalize();
		QVector2D point1ToMouse(point - lineStart);
		QVector2D point2ToMouse(point - lineEnd);
		double smallestDistance;
		if (point1ToMouse.length() * std::abs(QVector2D::dotProduct(pointConnection, point1ToMouse) / (pointConnection.length() * point1ToMouse.length())) > pointConnection.length()) {
			//perpendicular is not on line segment
			smallestDistance = ImageView::distance(lineEnd, point);
		} else if (point2ToMouse.length() * std::abs(QVector2D::dotProduct((-1)*pointConnection, point2ToMouse) / (pointConnection.length() * point2ToMouse.length())) > pointConnection.length()) {
			//perpendicular is also not on line segment
			smallestDistance = ImageView::distance(lineStart, point);
		} else {
			smallestDistance = std::abs(QVector2D::dotProduct(lineNormal, point1ToMouse));
		}
		return smallestDistance;
	}

	void ImageView::sharpen(cv::Mat& image, double strength, double radius) {
		cv::Mat tmp;
		cv::GaussianBlur(image, tmp, cv::Size(0, 0), radius);
		cv::addWeighted(image, 1 + strength, tmp, -strength, 0, image);
	}

	void ImageView::shallowCopyMatToImage(const cv::Mat& mat, QImage& destImage) {
		ImageView::matToImage(mat, destImage, false);
	}

	void ImageView::deepCopyMatToImage(const cv::Mat& mat, QImage& destImage) {
		ImageView::matToImage(mat, destImage, true);
	}

	void ImageView::shallowCopyImageToMat(const QImage& image, cv::Mat& destMat) {
		ImageView::imageToMat(image, destMat, false);
	}

	void ImageView::deepCopyImageToMat(const QImage& image, cv::Mat& destMat) {
		ImageView::imageToMat(image, destMat, true);
	}

	void ImageView::matToImage(const cv::Mat& mat, QImage& destImage, bool deepCopy) {
		if (mat.type() == CV_8UC4) {
			if (deepCopy) {
				destImage = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32).copy();
			} else {
				destImage = QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
			}
		} else if (mat.type() == CV_8UC3) {
			//this only works for opencv images that are RGB instead of GBR. Use cvtColor to change channel order.
			if (deepCopy) {
				destImage = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).copy();
			} else {
				destImage = QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
			}
		} else if (mat.type() == CV_8UC1) {
			QVector<QRgb> sColorTable;
			for (int i = 0; i < 256; ++i) {
				sColorTable.push_back(qRgb(i, i, i));
			}
			if (deepCopy) {
				destImage = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8).copy();
			} else {
				destImage = QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);
			}
			destImage.setColorTable(sColorTable);
		} else {
			std::cerr << "ERROR: Conversion from cv::Mat to QImage unsuccessfull because type is unknown." << std::endl;
		}
	}

	void ImageView::imageToMat(const QImage& image, cv::Mat& destMat, bool deepCopy) {
		if (image.format() == QImage::Format_ARGB32 || image.format() == QImage::Format_ARGB32_Premultiplied) {
			if (deepCopy) {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine()).clone();
			} else {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar*>(image.bits()), image.bytesPerLine());
			}
		} else if (image.format() == QImage::Format_RGB888) {
			if (deepCopy) {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine()).clone();
			} else {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
			}
		} else if (image.format() == QImage::Format_Indexed8) {
			if (deepCopy) {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC1, const_cast<uchar*>(image.bits()), image.bytesPerLine()).clone();
			} else {
				destMat = cv::Mat(image.height(), image.width(), CV_8UC1, const_cast<uchar*>(image.bits()), image.bytesPerLine());
			}
		} else {
			std::cerr << "ERROR: Conversion from QImage to cv::Mat unsuccessfull because type is unknown." << std::endl;
		}
	}

}