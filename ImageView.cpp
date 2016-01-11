#include "ImageView.h"

namespace hb {

	//========================================================================= Public =========================================================================\\

	ImageView::ImageView(QWidget *parent)
		: QWidget(parent),
		_interfaceOutline(true),
		_useHighQualityDownscaling(true),
		_rightClickForHundredPercentView(true),
		_usePanZooming(true),
		_imageAssigned(false),
		_isMat(false),
		_zoomBasis(1.5),
		_zoomExponent(0),
		_preventMagnificationInDefaultZoom(false),
		_hundredPercentZoomMode(false),
		_panOffset(0, 0),
		_viewRotation(0),
		_dragging(false),
		_pointEditingActive(false),
		_pointManipulationActive(false),
		_renderPoints(false),
		_moved(false),
		_panZooming(false),
		_paintingActive(false),
		_maskInitialized(false),
		_brushRadius(5),
		_brushPosition(0, 0),
		_painting(false),
		_visualizeBrushSize(false),
		_pointGrabTolerance(10),
		_pointGrabbed(false),
		_showPointDeletionWarning(false),
		_overlayMaskSet(false),
		_renderOverlayMask(false),
		_renderRectangle(false),
		_polylineAssigned(false),
		_renderPolyline(false),
		_useSmoothTransform(true),
		_enablePostResizeSharpening(false),
		_postResizeSharpeningStrength(0.5),
		_postResizeSharpeningRadius(1),
		_polylineManipulationActive(false),
		_polylinePointGrabbed(false),
		_polylineSelected(false),
		_polylinePointGrabTolerance(10),
		_polylineLastAddedPoint(0),
		_spanningSelectionRectangle(false),
		_polylineColor(60, 60, 60),
		_externalPostPaintFunctionAssigned(false) {
		setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
		setFocusPolicy(Qt::FocusPolicy::StrongFocus);
		setMouseTracking(true);
		QPalette palette = qApp->palette();
		_backgroundColor = palette.base().color();
	}

	QSize ImageView::sizeHint() const {
		return QSize(360, 360);
	}

	///Defines wether an outline should be drawn around the widget (indicating also if it has the focus or not)
	void ImageView::setShowInterfaceOutline(bool value) {
		_interfaceOutline = value;
	}

	///Sets the background colour or the widget.
	void ImageView::setInterfaceBackgroundColor(QColor const& color) {
		_backgroundColor = color;
	}

	///If set to true, a right click zooms the image to 100% magnification.
	void ImageView::setRightClickForHundredPercentView(bool value) {
		_rightClickForHundredPercentView = value;
	}

	///Returns \c true if the right click for 100% view feature is enabled.
	bool ImageView::rightClickForHundredPercentView() {
		return _rightClickForHundredPercentView;
	}

	///If set to true, panning while holding the middle mouse button will change the zoom.
	void ImageView::setUsePanZooming(bool value) {
		_usePanZooming = true;
	}

	///Returns true if pan-zooming is enabled.
	bool ImageView::usesPanZooming() {
		return _usePanZooming;
	}

	///Rotates the viewport 90° in anticlockwise direction.
	void ImageView::rotateLeft() {
		_viewRotation -= 90;
		if (_viewRotation < 0) _viewRotation += 360;
		enforcePanConstraints();
		updateResizedImage();
		if (isVisible()) update();
	}

	///Rotates the viewport 90° in clockwise direction.
	void ImageView::rotateRight() {
		_viewRotation += 90;
		if (_viewRotation >= 360) _viewRotation -= 360;
		enforcePanConstraints();
		updateResizedImage();
		if (isVisible()) update();
	}

	///Sets the rotation of the view to \p degrees degrees.
	void ImageView::setRotation(double degrees) {
		_viewRotation = degrees;
		if (_viewRotation >= 360) _viewRotation = _viewRotation - (360 * std::floor(degrees / 360.0));
		if (_viewRotation < 0) _viewRotation = _viewRotation + (360 * std::ceil(std::abs(degrees / 360.0)));
		enforcePanConstraints();
		updateResizedImage();
		if (isVisible()) update();
	}

	///Moves the viewport to the point \p point.
	void ImageView::centerViewportOn(QPointF point) {
		QPointF transformedPoint = getTransform().map(point);
		_panOffset += (QPointF((double)width() / 2.0, (double)height() / 2.0) - transformedPoint) / (pow(_zoomBasis, _zoomExponent)*getWindowScalingFactor());
		enforcePanConstraints();
		update();
	}

	///If set to true, images won't be enlarged at the default magnification (fit view).
	void ImageView::setPreventMagnificationInDefaultZoom(bool value) {
		_preventMagnificationInDefaultZoom = value;
		update();
	}

	///Makes the \c ImageView display the image \p image, shallow copy assignment.
	void ImageView::setImage(const QImage& image) {
		QSize oldSize = _image.size();
		_image = image;
		//free the mat
		_isMat = false;
		_mat = cv::Mat();
		if (_image.size() != oldSize) {
			resetMask();
			_hundredPercentZoomMode = false;
		}

		_imageAssigned = true;
		updateResizedImage();
		enforcePanConstraints();
		update();
	}

	///Makes the \c ImageView display the image \p image, move assignment.
	void ImageView::setImage(QImage&& image) {
		QSize oldSize = _image.size();
		_image = std::move(image);
		//free the mat
		_isMat = false;
		_mat = cv::Mat();
		if (_image.size() != oldSize) {
			resetMask();
			_hundredPercentZoomMode = false;
		}

		_imageAssigned = true;
		updateResizedImage();
		enforcePanConstraints();
		update();
	}

	///Makes the \c ImageView display the image \p image, shallow copy assignment.
	void ImageView::setImage(const cv::Mat& image) {
		if (image.type() == CV_8UC4 || image.type() == CV_8UC3 || image.type() == CV_8UC1) {
			QSize oldSize = _image.size();
			_mat = image;
			shallowCopyMatToImage(_mat, _image);
			if (_image.size() != oldSize) {
				resetMask();
				_hundredPercentZoomMode = false;
			}
			_isMat = true;

			_imageAssigned = true;
			updateResizedImage();
			enforcePanConstraints();
			update();
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
			QSize oldSize = _image.size();
			_mat = image;
			shallowCopyMatToImage(_mat, _image);
			if (_image.size() != oldSize) {
				resetMask();
				_hundredPercentZoomMode = false;
			}
			_downsampledMat = downscaledImage;
			shallowCopyMatToImage(_downsampledMat, _downsampledImage);
			_isMat = true;

			_imageAssigned = true;
			update();
		} else {
			std::cerr << "Image View: cannot assign image and downsampled preview because at least one is of unsupported type (" << image.type() << " and " << downscaledImage.type() << ")." << std::endl;
		}
	}

	///Removes the image.
	void ImageView::resetImage() {
		_image = QImage();
		_imageAssigned = false;
		update();
	}

	///Returns \c true if an image is assigned, false otherwise.
	bool ImageView::imageAssigned() const {
		return _imageAssigned;
	}

	///Returns the magnification factor at which the image is dispayed, 1 means the image is at a 100% view and one image pixel corresponds to one pixel of the display.
	double ImageView::getCurrentPreviewScalingFactor() const {
		if (_imageAssigned) {
			return std::pow(_zoomBasis, _zoomExponent) * getWindowScalingFactor();
		} else {
			return -1;
		}
	}

	///Specifies if the image will be resampled with a high quality algorithm when it's displayed with a magnificaiton smaller than 1.
	void ImageView::setUseHighQualityDownscaling(bool value) {
		_useHighQualityDownscaling = value;
		updateResizedImage();
		update();
	}

	///Returns \c true if high quality downscaling is enabled, \c false otherwise.
	bool ImageView::useHighQualityDownscaling() {
		return _useHighQualityDownscaling;
	}

	///Specifies if the sampling will be done bilinear or nearest neighbour when the iamge is displayed at a magnification greater than 1.
	void ImageView::setUseSmoothTransform(bool value) {
		_useSmoothTransform = value;
		update();
	}

	///Returns \c true if bilinear sampling is enabled, \c false otherwise.
	bool ImageView::useSmoothTransform() const {
		return _useSmoothTransform;
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
		_enablePostResizeSharpening = value;
		updateResizedImage();
		update();
	}

	///Returns \c true if the post resize sharpening is enabled, \c false otherwise.
	bool ImageView::enablePostResizeSharpening() {
		return _enablePostResizeSharpening;
	}

	///Sets the strength value of the post-resize unsharp masking filter to \p value.
	void ImageView::setPostResizeSharpeningStrength(double value) {
		_postResizeSharpeningStrength = value;
		updateResizedImage();
		update();
	}

	///Returns the strength value of the post-resize unsharp masking filter.
	double ImageView::postResizeSharpeningStrength() {
		return _postResizeSharpeningStrength;
	}

	///Sets the radius value of the post-resize unsharp masking filter to \p value.
	void ImageView::setPostResizeSharpeningRadius(double value) {
		_postResizeSharpeningRadius = value;
		updateResizedImage();
		update();
	}

	///Returns the radius value of the post-resize unsharp masking filter.
	double ImageView::postResizeSharpeningRadius() {
		return _postResizeSharpeningRadius;
	}

	///Sets all parameters for the post resize sharpening at once.
	/**
	* The advantage of using this function instead of setting the three
	* parameters separately is that the imageView will only have to update once,
	* resulting in better performance.
	*/
	void ImageView::setPostResizeSharpening(bool enable, double strength, double radius) {
		_enablePostResizeSharpening = enable;
		_postResizeSharpeningStrength = strength;
		_postResizeSharpeningRadius = radius;
		updateResizedImage();
		update();
	}

	///Enables or disables the ability to set new points and the ability to move already set ones; if adding points is enabled, manipulation of the polyline will be disabled.
	void ImageView::setPointEditing(bool enablePointAdding, bool enablePointManipulation) {
		_pointEditingActive = enablePointAdding;
		_pointManipulationActive = enablePointManipulation;
		if (enablePointAdding || enablePointManipulation)_renderPoints = true;
		if (enablePointAdding && _polylineManipulationActive) {
			std::cout << "Point adding was enabled, thus polyline manipulation will be disabled." << std::endl;
			_polylineManipulationActive = false;
		}
	}

	///Specpfies whether points are rendered or not.
	void ImageView::setRenderPoints(bool value) {
		_renderPoints = value;
		update();
	}

	///Returns the currently set points.
	const std::vector<QPointF>& ImageView::getPoints() const {
		return _points;
	}

	///Sets the points to \p points.
	void ImageView::setPoints(const std::vector<QPointF>& points) {
		_points = points;
		update();
	}

	///Sets the points to \p points.
	void ImageView::setPoints(std::vector<QPointF>&& points) {
		_points = std::move(points);
		update();
	}

	///Adds the point \p point.
	void ImageView::addPoint(const QPointF& point) {
		_points.push_back(point);
		update();
	}

	///Deletes all points which are outside the image, might happen when new image is assigned.
	void ImageView::deleteOutsidePoints() {
		for (std::vector<QPointF>::iterator point = _points.begin(); point != _points.end();) {
			if (point->x() < 0 || point->x() >= _image.width() || point->y() < 0 || point->y() >= _image.height()) {
				emit(userDeletedPoint(point - _points.begin()));
				point = _points.erase(point);
			} else {
				++point;
			}
		}
		update();
	}

	///Specifies whether it's possible to do overlay painting or not.
	/**
	 * Painting allows the user, for example, to mask certain areas.
	 */
	void ImageView::setPaintingActive(bool value) {
		if (value == true && !_maskInitialized && _imageAssigned) {
			_mask = QBitmap(_image.size());
			_mask.fill(Qt::color0);
			_maskInitialized = true;
		}
		if (value == false || _imageAssigned) {
			_paintingActive = value;
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
		_visualizeBrushSize = value;
		update();
	}

	///Returns the mask that has been painted by the user.
	const QBitmap& ImageView::getMask() const {
		return _mask;
	}

	///Sets a mask that will be displayed as a half-transparent overlay.
	/**
	 * This mask is not related to the panting the user does; instead it is an additional layer.
	 * This overload does a shallow copy assignment.
	 */
	void ImageView::setOverlayMask(const QBitmap& mask) {
		_overlayMask = mask;
		_overlayMaskSet = true;
		update();
	}

	///Sets a mask that will be displayed as a half-transparent overlay.
	/**
	* This mask is not related to the panting the user does; instead it is an additional layer.
	* This overload does a move assignment.
	*/
	void ImageView::setOverlayMask(QBitmap&& mask) {
		_overlayMask = std::move(mask);
		_overlayMaskSet = true;
		update();
	}

	///Specifies whether the assigned overlay mask is rendered or not.
	void ImageView::setRenderOverlayMask(bool value) {
		_renderOverlayMask = value;
		update();
	}

	void ImageView::setRenderRectangle(bool value) {
		_renderRectangle = value;
		update();
	}

	void ImageView::setRectangle(QRectF rectangle) {
		_rectangle = rectangle;
		update();
	}

	///Specifies whether the assigned polyline is rendered or not.
	void ImageView::setRenderPolyline(bool value) {
		_renderPolyline = value;
		update();
	}

	///Assigns a polyline that can be overlayed.
	void ImageView::setPolyline(std::vector<QPointF> border) {
		_polyline = border;
		_polylineAssigned = true;
		update();
	}

	///Enables or disables the ability to edit the polyline, will disable the ability to add points.
	void ImageView::setPolylineEditingActive(bool value) {
		_polylineManipulationActive = value;
		if (value && _pointEditingActive) {
			std::cout << "Polyline editing was enabled, thus point adding will be disabled." << std::endl;
			_pointEditingActive = false;
		}
	}

	///Returns the polyline as it currently is.
	const std::vector<QPointF>& ImageView::getPolyline() const {
		return _polyline;
	}

	///Sets the colour that the polyline is rendered in.
	void ImageView::setPolylineColor(QColor color) {
		_polylineColor = color;
		update();
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
		_externalPostPaint = function;
		_externalPostPaintFunctionAssigned = true;
	}

	//Removes any assigned post-paint function, which then is no longer invoked.
	void ImageView::removeExternalPostPaintFunction() {
		_externalPostPaintFunctionAssigned = false;
	}

	//========================================================================= Public Slots =========================================================================\\

	///Zooms the viewport in one step.
	void ImageView::zoomInKey() {
		QPointF center = QPointF(double(width()) / 2.0, double(height()) / 2.0);
		if (underMouse()) center = this->mapFromGlobal(QCursor::pos());
		zoomBy(120, center);
	}

	///Zooms the viewport out one step.
	void ImageView::zoomOutKey() {
		QPointF center = QPointF(double(width()) / 2.0, double(height()) / 2.0);
		if (underMouse()) center = this->mapFromGlobal(QCursor::pos());
		zoomBy(-120, center);
	}

	///Resets the mask the user is painting, does not affect the overlay mask.
	void ImageView::resetMask() {
		if (_maskInitialized) {
			_mask = QBitmap(_image.size());
			_mask.fill(Qt::color0);
			update();
		}
	}

	///Sets the radius of the brush to \p value.
	void ImageView::setBrushRadius(int value) {
		_brushRadius = value;
		if (_visualizeBrushSize) {
			update();
		}
	}

	///Displays the image at 100% magnification; the point \p center (in widget screen coordinates) will be centered.
	void ImageView::zoomToHundredPercent(QPointF center) {
		if (_imageAssigned) {
			QPointF mousePositionCoordinateBefore = getTransform().inverted().map(center);
			double desiredZoomFactor = 1 / getWindowScalingFactor();
			_zoomExponent = log(desiredZoomFactor) / log(_zoomBasis);
			QPointF mousePositionCoordinateAfter = getTransform().inverted().map(center);
			//remove the rotation from the delta
			QPointF mouseDelta = getTransformRotateOnly().map(mousePositionCoordinateAfter - mousePositionCoordinateBefore);
			_panOffset += mouseDelta;
			_hundredPercentZoomMode = true;
			enforcePanConstraints();
			updateResizedImage();
			update();
		}
	}

	void ImageView::resetZoom() {
		_zoomExponent = 0;
		_hundredPercentZoomMode = false;
		enforcePanConstraints();
		updateResizedImage();
		update();
	}

	///Deletes the point at index \p index.
	void ImageView::deletePoint(int index) {
		if (index >= 0 && index < _points.size()) {
			_points.erase(_points.begin() + index);
			update();
		}
	}

	///Removes all the set points.
	void ImageView::resetPoints() {
		_points.clear();
		update();
	}

	///Inverts the colour that the assigned polyline is rendered in.
	void ImageView::invertPolylineColor() {
		_polylineColor = QColor(255 - _polylineColor.red(), 255 - _polylineColor.green(), 255 - _polylineColor.blue());
		update();
	}

	//========================================================================= Protected =========================================================================\\

	void ImageView::showEvent(QShowEvent * e) {
		enforcePanConstraints();
	}

	void ImageView::mousePressEvent(QMouseEvent *e) {
		_lastMousePosition = e->pos();
		_initialMousePosition = e->pos();

		if (e->modifiers() & Qt::AltModifier && _polylineManipulationActive && _polylineAssigned) {
			//span a selection rectangle
			_polylineSelected = true;
			_selectionRectangle = QRectF(e->pos(), e->pos());
			if (!(e->modifiers() & Qt::ControlModifier)) {
				_polylineSelectedPoints.clear();
			}
			_spanningSelectionRectangle = true;
		} else {
			//check for close points to grab
			IndexWithDistance closestPoint = closestGrabbablePoint(e->pos());
			IndexWithDistance closestPolylinePoint = closestGrabbablePolylinePoint(e->pos());
			double polylineSelectionDistance = smallestDistanceToPolylineSelection(e->pos());
			if (closestPoint.index >= 0 && (closestPolylinePoint.index <= 0 || closestPoint.distance < closestPolylinePoint.distance || !_polylineSelected) && (closestPoint.distance < polylineSelectionDistance || _polylineSelectedPoints.size() == 0 || !_polylineSelected) && _pointManipulationActive) {
				//grab a point
				_grabbedPointIndex = closestPoint.index;
				qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
				_pointGrabbed = true;
			} else if ((closestPolylinePoint.index >= 0 || (polylineSelectionDistance <= _polylinePointGrabTolerance && _polylineSelectedPoints.size() > 0)) && _polylineManipulationActive && _polylineSelected) {
				//polyline editing
				if (((polylineSelectionDistance <= closestPolylinePoint.distance || _polylineSelectedPoints.find(closestPolylinePoint.index) != _polylineSelectedPoints.end()) && _polylineSelectedPoints.size() > 0) || closestPolylinePoint.index < 0 && !(e->modifiers() & Qt::ControlModifier) && !(e->modifiers() & Qt::ShiftModifier)) {
					//start moving the selection
					_polylinePointGrabbed = true;
				} else {
					if (closestPolylinePoint.index >= 0) {
						if (e->modifiers() & Qt::ShiftModifier) {
							if (!e->modifiers() & Qt::ControlModifier) _polylineSelectedPoints.clear();
							//add all points inbetween the current point and the last point
							int largerIndex = std::max(closestPolylinePoint.index, _polylineLastAddedPoint);
							int smallerIndex = std::min(closestPolylinePoint.index, _polylineLastAddedPoint);
							for (int index = smallerIndex; index <= largerIndex; ++index) {
								_polylineSelectedPoints.insert(index);
							}
						} else {
							if (e->modifiers() & Qt::ControlModifier) {
								//add point to selected points or remove it
								std::set<int>::iterator point = _polylineSelectedPoints.find(closestPolylinePoint.index);
								if (point == _polylineSelectedPoints.end()) {
									_polylineSelectedPoints.insert(closestPolylinePoint.index);
								} else {
									_polylineSelectedPoints.erase(point);
								}
								_polylineLastAddedPoint = closestPolylinePoint.index;
							} else {
								_polylineSelectedPoints.clear();
								//grab polyline point
								_polylineSelectedPoints.insert(closestPolylinePoint.index);
								_polylineLastAddedPoint = closestPolylinePoint.index;
								_polylinePointGrabbed = true;
							}
						}
					}
				}
			} else if ((!_paintingActive && e->button() != Qt::MiddleButton) || (_paintingActive && e->button() == Qt::MiddleButton)) {
				//dragging
				_dragging = true;		
			} else if (e->button() == Qt::MiddleButton) {
				//pan-zooming
				_panZooming = true;
				_panZoomingInitialPanOffset = _panOffset;
				_panZoomingInitialZoomExponent = _zoomExponent;
				qApp->setOverrideCursor(QCursor(Qt::SizeVerCursor));
			} else if (_imageAssigned) {
				//painting
				_painting = true;

				//paint a circle
				QPainter canvas(&_mask);
				canvas.setPen(Qt::NoPen);
				if (e->button() == Qt::LeftButton) {
					canvas.setBrush(QBrush(Qt::color1));
				} else {
					canvas.setBrush(QBrush(Qt::color0));
				}
				QTransform transform = getTransform().inverted();
				canvas.drawEllipse(transform.map(QPointF(e->pos())), _brushRadius, _brushRadius);
				update();
			}
		}

		_moved = false;
	}

	void ImageView::mouseMoveEvent(QMouseEvent *e) {
		_moved = true;

		if (_dragging || _pointGrabbed || _polylinePointGrabbed) {
			QPointF deltaRotated = getTransformScaleRotateOnly().inverted().map((e->pos() - _lastMousePosition));
			QPointF deltaScaled = getTransformScaleOnly().inverted().map((e->pos() - _lastMousePosition));
			if (_dragging) {
				//dragging
				qApp->setOverrideCursor(QCursor(Qt::ClosedHandCursor));
				_panOffset += deltaScaled;
				enforcePanConstraints();
			} else if (_pointGrabbed) {
				//editing points
				_points[_grabbedPointIndex] += deltaRotated;
				if (e->pos().x() < 0 || e->pos().y() < 0 || e->pos().x() > width() || e->pos().y() > height() || _points[_grabbedPointIndex].x() < 0 || _points[_grabbedPointIndex].y() < 0 || _points[_grabbedPointIndex].x() >= _image.width() || _points[_grabbedPointIndex].y() >= _image.height()) {
					_showPointDeletionWarning = true;
					qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
				} else {
					_showPointDeletionWarning = false;
					qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
				}
				emit pointModified();
			} else {
				//editing polyline points
				for (int index : _polylineSelectedPoints) {
					_polyline[index] += deltaRotated;
					if (_polyline[index].x() < 0)_polyline[index].setX(0);
					if (_polyline[index].x() > _image.width())_polyline[index].setX(_image.width());
					if (_polyline[index].y() < 0)_polyline[index].setY(0);
					if (_polyline[index].y() > _image.height())_polyline[index].setY(_image.height());
				}
			}
			update();
		} else if (_spanningSelectionRectangle) {
			_selectionRectangle.setBottomLeft(e->pos());
			QTransform transform = getTransform();
			_selectionRectanglePoints.clear();
			for (int point = 0; point < _polyline.size(); ++point) {
				QPointF transformedPoint = transform.map(_polyline[point]);
				if (_selectionRectangle.contains(transformedPoint)) {
					_selectionRectanglePoints.insert(point);
					_polylineLastAddedPoint = point;
				}
			}
			update();
		}

		if (_paintingActive) {
			_brushPosition = e->pos();
			if (_painting) {
				//draw a line from last mouse position to the current
				QPainter canvas(&_mask);
				QPen pen;
				if (e->buttons() == Qt::LeftButton) {
					pen.setColor(Qt::color1);
				} else {
					pen.setColor(Qt::color0);
				}
				pen.setWidth(2 * _brushRadius);
				pen.setCapStyle(Qt::RoundCap);
				canvas.setPen(pen);
				QTransform transform = getTransform().inverted();
				canvas.drawLine(transform.map(_lastMousePosition), transform.map(e->pos()));
			}
			update();
		}

		if (_panZooming) {
			_zoomExponent = _panZoomingInitialZoomExponent;
			_panOffset = _panZoomingInitialPanOffset;
			double delta = (_initialMousePosition - e->pos()).y() * (-3);
			zoomBy(delta, _initialMousePosition);
			//doesn't work as expected
			//QCursor::setPos(mapToGlobal(_lastMousePosition.toPoint()));
		}

		if (!_dragging && !_painting && !_pointGrabbed && !_spanningSelectionRectangle && !_panZooming) {
			//check for close points to grab
			if (_pointManipulationActive) {
				if (closestGrabbablePoint(e->pos()).index >= 0) {
					qApp->setOverrideCursor(QCursor(Qt::OpenHandCursor));
				} else {
					qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
				}
			}
		}

		if (_dragging || _painting || _pointGrabbed || _polylinePointGrabbed) {
			_lastMousePosition = e->pos();
		}
	}

	void ImageView::mouseReleaseEvent(QMouseEvent *e) {

		//clicking points
		if (_pointEditingActive && _imageAssigned && !_moved) {
			//this was a click, add a point
			QTransform transform = getTransform();
			QPointF clickedPoint = e->pos();
			QPointF worldPoint = transform.inverted().map(clickedPoint);
			if (worldPoint.x() >= 0 && worldPoint.x() <= _image.width() && worldPoint.y() >= 0 && worldPoint.y() <= _image.height()) {
				_points.push_back(worldPoint);
				std::cout << "Point added: " << worldPoint.x() << "  " << worldPoint.y() << std::endl;
				emit pointModified();
			}
		} else if (!_pointEditingActive && !_moved && _imageAssigned) {
			if (_polylineManipulationActive && e->button() != Qt::RightButton) {
				//this was a click, select or unselect polyline
				if (smallestDistanceToPolyline(e->pos()) <= _polylinePointGrabTolerance) {
					//clicked close enough to a point, select line
					_polylineSelected = true;
				} else {
					//clicked somewehere else, deselect it
					_polylineSelected = false;
					_polylineSelectedPoints.clear();
				}
			}

			if (e->button() == Qt::RightButton && _rightClickForHundredPercentView) {
				//zoom to 100%
				if (_hundredPercentZoomMode) {
					resetZoom();
				} else {
					zoomToHundredPercent(e->pos());
				}
			}

			//emit pixel click signal
			QTransform transform = getTransform();
			QPointF clickedPoint = e->pos();
			QPointF worldPoint = transform.inverted().map(clickedPoint);
			emit(pixelClicked(QPoint(std::floor(worldPoint.x()), std::floor(worldPoint.y()))));
		}

		if (_pointGrabbed) {
			if (e->pos().x() < 0 || e->pos().y() < 0 || e->pos().x() > width() || e->pos().y() > height() || _points[_grabbedPointIndex].x() < 0 || _points[_grabbedPointIndex].y() < 0 || _points[_grabbedPointIndex].x() >= _image.width() || _points[_grabbedPointIndex].y() >= _image.height()) {
				deletePoint(_grabbedPointIndex);
				emit(userDeletedPoint(_grabbedPointIndex));
				_showPointDeletionWarning = false;
				qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			} else {
				qApp->setOverrideCursor(QCursor(Qt::OpenHandCursor));
			}
			_pointGrabbed = false;
			emit pointModified();
		}

		if (_polylinePointGrabbed) {
			_polylinePointGrabbed = false;
			if (_moved) emit polylineModified();
		}

		if (_spanningSelectionRectangle) {
			_spanningSelectionRectangle = false;
			_polylineSelectedPoints.insert(_selectionRectanglePoints.begin(), _selectionRectanglePoints.end());
			_selectionRectanglePoints.clear();
		}

		if (_dragging) {
			if (_paintingActive) {
				qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
			} else {
				qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			}
			_dragging = false;
		}
		_painting = false;

		if (_panZooming) {
			qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
			_panZooming = false;
		}

		update();
	}

	void ImageView::mouseDoubleClickEvent(QMouseEvent* e) {
		e->ignore();
	}

	void ImageView::wheelEvent(QWheelEvent* e) {
		if (!_panZooming) {
			zoomBy(e->delta(), e->pos(), e->modifiers());
		}
		e->accept();
	}

	void ImageView::resizeEvent(QResizeEvent* e) {
		//maintain 100% view if in 100% view
		if (_hundredPercentZoomMode) {
			QPointF center(width() / 2.0, height() / 2.0);
			zoomToHundredPercent(center);
		}
		updateResizedImage();
	}

	void ImageView::enterEvent(QEvent* e) {
		if (_paintingActive) {
			qApp->setOverrideCursor(QCursor(Qt::BlankCursor));
		}
	}

	void ImageView::leaveEvent(QEvent* e) {
		qApp->setOverrideCursor(QCursor(Qt::ArrowCursor));
		if (_paintingActive) {
			update();
		}
	}

	void ImageView::paintEvent(QPaintEvent* e) {
		QPainter canvas(this);
		canvas.setRenderHint(QPainter::Antialiasing, true);
		canvas.setRenderHint(QPainter::SmoothPixmapTransform, _useSmoothTransform);
		QSize canvasSize = size();
		QTransform transform = getTransform();
		QPalette palette = qApp->palette();
		canvas.fillRect(0, 0, width(), height(), _backgroundColor);

		//drawing of the image
		if (_imageAssigned) {
			if (std::pow(_zoomBasis, _zoomExponent) * getWindowScalingFactor() >= 1 || !_useHighQualityDownscaling) {
				canvas.setTransform(transform);
				canvas.drawImage(QPoint(0, 0), _image);
			} else {
				canvas.setTransform(getTransformDownsampledImage());
				canvas.drawImage(QPoint(0, 0), _downsampledImage);
			}
		}

		//drawing of the overlay mask
		if (_overlayMaskSet && _renderOverlayMask) {
			canvas.setTransform(transform);
			QImage image = _overlayMask.toImage();
			image.setColor(Qt::color0, QColor(Qt::white).rgb());
			image.setColor(Qt::color1, Qt::transparent);
			canvas.setOpacity(0.9);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, false);
			canvas.drawImage(QPoint(0, 0), image);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, true);
			canvas.setOpacity(1);
		}

		//drawing of bounds (rectangle) overlay
		if (_imageAssigned && _renderRectangle) {
			QPixmap rect = QPixmap(canvasSize);
			rect.fill(Qt::transparent);
			QRectF imageArea(QPointF(0, 0), _image.size());
			imageArea = transform.mapRect(imageArea);
			QPainter p(&rect);
			p.setRenderHint(QPainter::Antialiasing, true);
			p.setPen(Qt::NoPen);
			p.setBrush(QColor(0, 0, 0, 100));
			p.drawRect(imageArea);
			p.setBrush(QBrush(Qt::transparent));
			p.setCompositionMode(QPainter::CompositionMode_SourceOut);
			QRectF eraseRect = transform.mapRect(_rectangle);
			p.drawRect(eraseRect);
			canvas.resetTransform();
			canvas.drawPixmap(0, 0, rect);
		}

		//drawing of the mask that is currently painted
		if (_paintingActive) {
			canvas.setTransform(transform);
			QImage image = _mask.toImage();
			image.setColor(Qt::color0, Qt::transparent);
			image.setColor(Qt::color1, QColor(Qt::red).rgb());
			canvas.setOpacity(0.5);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, false);
			canvas.drawImage(QPoint(0, 0), image);
			canvas.setRenderHint(QPainter::SmoothPixmapTransform, true);
			canvas.setOpacity(1);
		}

		//drawing of the polyline if assigned
		if (_polylineAssigned && _renderPolyline && _polyline.size() > 0) {
			canvas.setRenderHint(QPainter::Antialiasing, true);
			canvas.setTransform(transform);
			QPen linePen = QPen(_polylineColor);
			linePen.setJoinStyle(Qt::MiterJoin);
			if (!_polylineSelected || !_polylineManipulationActive) linePen.setWidth(3);
			QBrush brush = QBrush(_polylineColor);
			linePen.setCosmetic(true);
			canvas.setPen(linePen);
			if (_polyline.size() > 1) {
				canvas.drawPolyline(_polyline.data(), _polyline.size());
				if (_polylineManipulationActive && _polylineSelected) {
					canvas.resetTransform();
					const int squareSize = 4;
					const int squareOffset = squareSize / 2;
					for (int point = 0; point < _polyline.size(); ++point) {
						if (_selectionRectanglePoints.find(point) != _selectionRectanglePoints.end() || _polylineSelectedPoints.find(point) != _polylineSelectedPoints.end()) {
							canvas.setBrush(brush);
						} else {
							canvas.setBrush(Qt::NoBrush);
						}
						QPointF transformedPoint = transform.map(_polyline[point]);
						canvas.drawRect(transformedPoint.x() - squareOffset, transformedPoint.y() - squareOffset, squareSize, squareSize);
					}
				}
			} else {
				canvas.drawPoint(_polyline[0]);
			}
		}

		//draw selection rectangle when selecting
		if (_spanningSelectionRectangle) {
			canvas.resetTransform();
			canvas.setPen(QPen(Qt::darkGray, 1, Qt::DashDotLine));
			canvas.setBrush(Qt::NoBrush);
			canvas.drawRect(_selectionRectangle);
		}

		//drawing of the points
		if (_renderPoints) {
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
			for (int point = 0; point < _points.size(); ++point) {
				transformedPoint = transform.map(_points[point]);
				canvas.setPen(pen);
				canvas.drawEllipse(transformedPoint, 5, 5);
				canvas.setPen(textPen);
				canvas.drawText(transformedPoint + QPointF(7.0, 14.0), QString::number(point + 1));
			}
			if (_pointEditingActive) {
				canvas.setPen(textPen);
				QString statusMessage = ((_points.size() != 1) ? QString(tr("There are ")) : QString(tr("There is "))) + QString::number(_points.size()) + ((_points.size() != 1) ? QString(tr(" points set.")) : QString(tr(" point set.")));
				canvas.drawText(QPoint(20, height() - 15), statusMessage);
			}
		}

		//if painting active draw brush outline
		if (_paintingActive && underMouse() && !_dragging) {
			canvas.resetTransform();
			double scalingFactor = pow(_zoomBasis, _zoomExponent) * getWindowScalingFactor();
			canvas.setBrush(Qt::NoBrush);
			canvas.setPen(QPen(Qt::darkGray, 1));
			canvas.drawEllipse(_brushPosition, _brushRadius*scalingFactor, _brushRadius*scalingFactor);
		}

		//visualization of the brush size (e.g. when changing it)
		if (_visualizeBrushSize) {
			canvas.resetTransform();
			canvas.setPen(QPen(Qt::darkGray));
			canvas.setBrush(Qt::NoBrush);
			double scalingFactor = pow(_zoomBasis, _zoomExponent) * getWindowScalingFactor();
			canvas.drawEllipse(QPointF((double)width() / 2.0, (double)height() / 2.0), _brushRadius*scalingFactor, _brushRadius*scalingFactor);
		}

		//the point deletion warning
		if (_showPointDeletionWarning) {
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
			canvas.drawText(QRect(0, 0, width(), height()), Qt::AlignCenter, QString(tr("Release to delete point")));
		}

		//add a contour
		if (_interfaceOutline) {
			canvas.resetTransform();
			canvas.setRenderHint(QPainter::Antialiasing, 0);
			QColor strokeColour;
			if (hasFocus()) {
				strokeColour = palette.highlight().color();
			} else {
				strokeColour = palette.base().color();
				strokeColour.setRed(strokeColour.red() / 2);
				strokeColour.setGreen(strokeColour.green() / 2);
				strokeColour.setBlue(strokeColour.blue() / 2);
			}
			canvas.setPen(QPen(strokeColour, 1));
			canvas.setBrush(Qt::NoBrush);
			canvas.drawRect(0, 0, width() - 1, height() - 1);
		}

		//call external post paint function
		if (_externalPostPaintFunctionAssigned) {
			canvas.resetTransform();
			_externalPostPaint(canvas);
		}
	}

	void ImageView::keyPressEvent(QKeyEvent * e) {
		if ((isVisible() && (underMouse() || e->key() == Qt::Key_X) && _imageAssigned) || e->key() == Qt::Key_S) {
			if (e->key() == Qt::Key_Plus && !_panZooming) {
				zoomInKey();
			} else if (e->key() == Qt::Key_Minus && !_panZooming) {
				zoomOutKey();
			} else if (e->key() == Qt::Key_S) {
				setUseSmoothTransform(!_useSmoothTransform);
			} else if (e->key() == Qt::Key_X && _polylineAssigned && _renderPolyline) {
				invertPolylineColor();
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
			if ((keyEvent->key() == Qt::Key_Plus || keyEvent->key() == Qt::Key_Minus) && isVisible() && underMouse() && _imageAssigned) {
				keyPressEvent(keyEvent);
				return true;
			} else if (keyEvent->key() == Qt::Key_S) {
				keyPressEvent(keyEvent);
			} else if (keyEvent->key() == Qt::Key_X && isVisible() && _imageAssigned && _polylineAssigned && _renderPolyline) {
				keyPressEvent(keyEvent);
				return true;
			}
		}
		return false;
	}

	//========================================================================= Private =========================================================================\\

	double ImageView::getEffectiveImageWidth() const {
		return std::abs(std::cos(_viewRotation * M_PI / 180)) * (double)_image.width() + std::abs(std::sin(_viewRotation * M_PI / 180)) * (double)_image.height();
	}

	double ImageView::getEffectiveImageHeight() const {
		return std::abs(std::cos(_viewRotation * M_PI / 180)) * (double)_image.height() + std::abs(std::sin(_viewRotation * M_PI / 180)) * (double)_image.width();
	}

	double ImageView::getWindowScalingFactor() const {
		if (_imageAssigned && _image.width() != 0 && _image.height() != 0) {
			double imageWidth = getEffectiveImageWidth();
			double imageHeight = getEffectiveImageHeight();
			double scalingFactor = std::min((double)size().width() / imageWidth, (double)size().height() / imageHeight);
			if (_preventMagnificationInDefaultZoom && scalingFactor > 1) {
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
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		double centeringOffsetX = (double)_image.width() / 2;
		double centeringOffsetY = (double)_image.height() / 2;
		double transX = ((width() / factor) - _image.width()) / 2;
		double transY = ((height() / factor) - _image.height()) / 2;
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
		transform.translate(_panOffset.x(), _panOffset.y());
		//rotate the view
		transform.rotate(_viewRotation);
		//move image center to 0, 0
		transform.translate((-1)*centeringOffsetX, (-1)*centeringOffsetY);

		return transform;
	}

	QTransform ImageView::getTransformDownsampledImage() const {
		//makes the map always fill the whole interface element
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		/*Here we can do integer division for the centering offset because this function is only called
		when the image is displayed with negative magnificaiton. The Error of ca. 0.5 pixels can be
		accepted in this case because it will not be visible very much. Floating point numbers in
		contrast whould result in a slightly blurred image when the image is rotated and one ofset
		is integer while the other one is a fraction (because of the difference when moving the image to
		the origin and moving the image back would be < 1px due to the intermediate roation)*/
		double centeringOffsetX = _downsampledImage.width() / 2;
		double centeringOffsetY = _downsampledImage.height() / 2;
		double transX = ((width()) - _downsampledImage.width()) / 2;
		double transY = ((height()) - _downsampledImage.height()) / 2;
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
		transform.translate(_panOffset.x() * zoomFactor * factor, _panOffset.y() * zoomFactor * factor);
		//rotate the view
		transform.rotate(_viewRotation);
		//move image center to 0, 0
		transform.translate((-1)*centeringOffsetX, (-1)*centeringOffsetY);

		return transform;
	}

	QTransform ImageView::getTransformScaleRotateOnly() const {
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		transform.scale(factor, factor);
		//apply users zoom
		transform.scale(zoomFactor, zoomFactor);
		//rotate the view
		transform.rotate(_viewRotation);
		return transform;
	}

	QTransform ImageView::getTransformScaleOnly() const {
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//apply the window scaling factor
		transform.scale(factor, factor);
		//apply users zoom
		transform.scale(zoomFactor, zoomFactor);
		return transform;
	}

	QTransform ImageView::getTransformRotateOnly() const {
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		//those transforms are performed in inverse order, so read bottom - up
		QTransform transform;
		//rotate the view
		transform.rotate(_viewRotation);
		return transform;
	}

	void ImageView::zoomBy(double delta, QPointF const& center, Qt::KeyboardModifiers modifier) {
		if (_imageAssigned) {
			QPointF mousePositionCoordinateBefore = getTransform().inverted().map(center);
			if (modifier & Qt::ControlModifier) {
				_zoomExponent += delta / 600;
			} else if (!modifier) {
				_zoomExponent += delta / 120;
			} else {
				return;
			}
			if (_zoomExponent < 0)_zoomExponent = 0;
			QPointF mousePositionCoordinateAfter = getTransform().inverted().map(center);
			//remove the rotation from the delta
			QPointF mouseDelta = getTransformRotateOnly().map(mousePositionCoordinateAfter - mousePositionCoordinateBefore);
			_panOffset += mouseDelta;
			_hundredPercentZoomMode = false;
			enforcePanConstraints();
			updateResizedImage();
			update();
		}
	}

	void ImageView::enforcePanConstraints() {
		double imageWidth = getEffectiveImageWidth();
		double imageHeight = getEffectiveImageHeight();
		double factor = getWindowScalingFactor();
		double zoomFactor = pow(_zoomBasis, _zoomExponent);
		double maxXOffset = (-1)*(((width() / factor / zoomFactor) - imageWidth) / 2);
		double maxYOffset = (-1)*(((height() / factor / zoomFactor) - imageHeight) / 2);
		maxXOffset = std::max(0.0, maxXOffset);
		maxYOffset = std::max(0.0, maxYOffset);
		if (_panOffset.x() > maxXOffset)_panOffset.setX(maxXOffset);
		if (_panOffset.x() < (-1) * maxXOffset)_panOffset.setX((-1) * maxXOffset);
		if (_panOffset.y() > maxYOffset)_panOffset.setY(maxYOffset);
		if (_panOffset.y() < (-1) * maxYOffset)_panOffset.setY((-1) * maxYOffset);
	}

	void ImageView::updateResizedImage() {
		if (_useHighQualityDownscaling && _imageAssigned) {
			double scalingFactor = std::pow(_zoomBasis, _zoomExponent) * getWindowScalingFactor();
			if (scalingFactor < 1) {
				if (!_isMat) {
					if (_image.format() == QImage::Format_RGB888 || _image.format() == QImage::Format_Indexed8 || _image.format() == QImage::Format_ARGB32) {
						cv::Mat orig;
						shallowCopyImageToMat(_image, orig);
						cv::resize(orig, _downsampledMat, cv::Size(), scalingFactor, scalingFactor, cv::INTER_AREA);
						if (_enablePostResizeSharpening) {
							ImageView::sharpen(_downsampledMat, _postResizeSharpeningStrength, _postResizeSharpeningRadius);
						}
						deepCopyMatToImage(_downsampledMat, _downsampledImage);
					} else {
						//alternative
						_downsampledImage = _image.scaledToWidth(_image.width() * scalingFactor, Qt::SmoothTransformation);
					}
				} else {
					cv::resize(_mat, _downsampledMat, cv::Size(), scalingFactor, scalingFactor, cv::INTER_AREA);
					if (_enablePostResizeSharpening) {
						ImageView::sharpen(_downsampledMat, _postResizeSharpeningStrength, _postResizeSharpeningRadius);
					}
					shallowCopyMatToImage(_downsampledMat, _downsampledImage);
				}
			}
		}
	}

	double ImageView::distance(const QPointF& point1, const QPointF& point2) {
		return std::sqrt(std::pow(point2.x() - point1.x(), 2) + std::pow(point2.y() - point1.y(), 2));
	}

	ImageView::IndexWithDistance ImageView::closestGrabbablePoint(QPointF const& mousePosition) const {
		if (_points.size() > 0) {
			QTransform transform = getTransform();
			double smallestDistance = distance(transform.map(_points[0]), mousePosition);
			double index = 0;
			for (int point = 1; point < _points.size(); ++point) {
				double tmpDistance = distance(transform.map(_points[point]), mousePosition);
				if (tmpDistance < smallestDistance) {
					smallestDistance = tmpDistance;
					index = point;
				}
			}
			if (smallestDistance < _pointGrabTolerance) {
				return IndexWithDistance(index, smallestDistance);
			}
		}
		return IndexWithDistance(-1, 0);
	}

	ImageView::IndexWithDistance ImageView::closestGrabbablePolylinePoint(QPointF const& mousePosition) const {
		if (_polyline.size() > 0) {
			QTransform transform = getTransform();
			double smallestDistance = distance(transform.map(_polyline[0]), mousePosition);
			double index = 0;
			for (int point = 1; point < _polyline.size(); ++point) {
				double tmpDistance = distance(transform.map(_polyline[point]), mousePosition);
				if (tmpDistance < smallestDistance) {
					smallestDistance = tmpDistance;
					index = point;
				}
			}
			if (smallestDistance < _pointGrabTolerance) {
				return IndexWithDistance(index, smallestDistance);
			}
		}
		return IndexWithDistance(-1, 0);
	}

	double ImageView::smallestDistanceToPolyline(QPointF const& mousePosition) const {
		if (_polyline.size() > 0) {
			QTransform transform = getTransform();
			double smallestDistance = distance(_polyline[0], mousePosition);
			if (_polyline.size() > 1) {
				for (int point = 0; point < _polyline.size() - 1; ++point) {
					QPointF point1 = transform.map(_polyline[point]);
					QPointF point2 = transform.map(_polyline[point + 1]);
					double d = distanceOfPointToLineSegment(point1, point2, mousePosition);
					if (d < smallestDistance) smallestDistance = d;
				}
			}
			return smallestDistance;
		}
		return 0;
	}

	double ImageView::smallestDistanceToPolylineSelection(QPointF const& mousePosition) const {
		if (_polyline.size() > 0) {
			QTransform transform = getTransform();
			double smallestDistance = -1;
			for (int index : _polylineSelectedPoints) {
				QPointF point1 = transform.map(_polyline[index]);
				double d;
				if (_polylineSelectedPoints.find(index + 1) != _polylineSelectedPoints.end()) {
					//check distance to line segment	
					QPointF point2 = transform.map(_polyline[index + 1]);
					d = distanceOfPointToLineSegment(point1, point2, mousePosition);
				} else {
					//check distance to point
					d = distance(point1, mousePosition);
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
			smallestDistance = distance(lineEnd, point);
		} else if (point2ToMouse.length() * std::abs(QVector2D::dotProduct((-1)*pointConnection, point2ToMouse) / (pointConnection.length() * point2ToMouse.length())) > pointConnection.length()) {
			//perpendicular is also not on line segment
			smallestDistance = distance(lineStart, point);
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
		matToImage(mat, destImage, false);
	}

	void ImageView::deepCopyMatToImage(const cv::Mat& mat, QImage& destImage) {
		matToImage(mat, destImage, true);
	}

	void ImageView::shallowCopyImageToMat(const QImage& image, cv::Mat& destMat) {
		imageToMat(image, destMat, false);
	}

	void ImageView::deepCopyImageToMat(const QImage& image, cv::Mat& destMat) {
		imageToMat(image, destMat, true);
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