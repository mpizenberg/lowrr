port module Main exposing (main)

import Browser
import CropForm
import Device exposing (Device)
import Dict exposing (Dict)
import Element exposing (Element, alignRight, centerX, centerY, fill, height, padding, paddingXY, spacing, width)
import Element.Background
import Element.Border
import Element.Font
import Element.Input
import FileValue as File exposing (File)
import Html exposing (Html)
import Html.Attributes
import Html.Events
import Html.Events.Extra.Pointer as Pointer
import Html.Events.Extra.Wheel as Wheel
import Icon
import Json.Decode exposing (Decoder, Value)
import Keyboard exposing (RawKey)
import NumberInput
import Pivot exposing (Pivot)
import Set exposing (Set)
import Simple.Transition as Transition
import Style
import Svg
import Svg.Attributes
import Viewer exposing (Viewer)
import Viewer.Svg


port resizes : (Device.Size -> msg) -> Sub msg


port decodeImages : List Value -> Cmd msg


port imageDecoded : (Image -> msg) -> Sub msg


port capture : Value -> Cmd msg


main : Program Device.Size Model Msg
main =
    Browser.element
        { init = init
        , view = view
        , update = update
        , subscriptions = subscriptions
        }


type alias Model =
    -- Current state of the application
    { state : State
    , device : Device
    , params : Parameters
    , paramsForm : ParametersForm
    , paramsInfo : ParametersToggleInfo
    , viewer : Viewer
    , pointerMode : PointerMode
    , bboxDrawn : Maybe BBox
    }


type alias BBox =
    { left : Float
    , top : Float
    , right : Float
    , bottom : Float
    }


type State
    = Home FileDraggingState
    | Loading { names : Set String, loaded : Dict String Image }
    | ViewImgs { images : Pivot Image }
    | Config { images : Pivot Image }
    | Registration { images : Pivot Image }
    | Logs { images : Pivot Image }


type FileDraggingState
    = Idle
    | DraggingSomeFiles


type alias Image =
    { id : String
    , url : String
    , width : Int
    , height : Int
    }


type alias Parameters =
    { crop : Maybe Crop
    , equalize : Bool
    , levels : Int
    , sparse : Float
    , lambda : Float
    , rho : Float
    , maxIterations : Int
    , convergenceThreshold : Float
    }


type alias Crop =
    { left : Int
    , top : Int
    , right : Int
    , bottom : Int
    }


type alias ParametersForm =
    { crop : CropForm.State
    , maxIterations : NumberInput.Field Int NumberInput.IntError
    , convergenceThreshold : NumberInput.Field Float NumberInput.FloatError
    , levels : NumberInput.Field Int NumberInput.IntError
    , sparse : NumberInput.Field Float NumberInput.FloatError
    , lambda : NumberInput.Field Float NumberInput.FloatError
    , rho : NumberInput.Field Float NumberInput.FloatError
    }


type alias ParametersToggleInfo =
    { crop : Bool
    , maxIterations : Bool
    , convergenceThreshold : Bool
    , levels : Bool
    , sparse : Bool
    , lambda : Bool
    , rho : Bool
    }


type PointerMode
    = WaitingMove
    | PointerMovingFromClientCoords ( Float, Float )
    | WaitingDraw
    | PointerDrawFromOffsetAndClient ( Float, Float ) ( Float, Float )


{-| Initialize the model.
-}
init : Device.Size -> ( Model, Cmd Msg )
init size =
    ( { state = initialState
      , device = Device.classify size
      , params = defaultParams
      , paramsForm = defaultParamsForm
      , paramsInfo = defaultParamsInfo
      , viewer = Viewer.withSize ( size.width, size.height - toFloat headerHeight )
      , pointerMode = WaitingMove
      , bboxDrawn = Nothing
      }
    , Cmd.none
    )


initialState : State
initialState =
    -- Home Idle
    -- Config { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }
    -- ViewImgs { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }
    ViewImgs { images = Pivot.fromCons (Image "header" "/img/pano_bayeux.jpg" 2000 225) [] }


defaultParams : Parameters
defaultParams =
    { crop = Nothing
    , equalize = True
    , levels = 4
    , sparse = 0.5
    , lambda = 1.5
    , rho = 0.1
    , maxIterations = 40
    , convergenceThreshold = 0.001
    }


defaultParamsForm : ParametersForm
defaultParamsForm =
    let
        anyInt =
            NumberInput.intDefault

        anyFloat =
            NumberInput.floatDefault
    in
    { crop = CropForm.withSize 1920 1080
    , maxIterations =
        { anyInt | min = Just 1, max = Just 1000 }
            |> NumberInput.setDefaultIntValue defaultParams.maxIterations
    , convergenceThreshold =
        { defaultValue = defaultParams.convergenceThreshold
        , min = Just 0.0
        , max = Nothing
        , increase = \x -> x * sqrt 2
        , decrease = \x -> x / sqrt 2
        , input = String.fromFloat defaultParams.convergenceThreshold
        , decodedInput = Ok defaultParams.convergenceThreshold
        }
    , levels =
        { anyInt | min = Just 1, max = Just 10 }
            |> NumberInput.setDefaultIntValue defaultParams.levels
    , sparse =
        { anyFloat | min = Just 0.0, max = Just 1.0 }
            |> NumberInput.setDefaultFloatValue defaultParams.sparse
    , lambda =
        { anyFloat | min = Just 0.0 }
            |> NumberInput.setDefaultFloatValue defaultParams.lambda
    , rho =
        { defaultValue = defaultParams.rho
        , min = Just 0.0
        , max = Nothing
        , increase = \x -> x * sqrt 2
        , decrease = \x -> x / sqrt 2
        , input = String.fromFloat defaultParams.rho
        , decodedInput = Ok defaultParams.rho
        }
    }


defaultParamsInfo : ParametersToggleInfo
defaultParamsInfo =
    { crop = False
    , maxIterations = False
    , convergenceThreshold = False
    , levels = False
    , sparse = False
    , lambda = False
    , rho = False
    }



-- Update ############################################################


type Msg
    = NoMsg
    | WindowResizes Device.Size
    | DragDropMsg DragDropMsg
    | ImageDecoded Image
    | KeyDown RawKey
    | ZoomMsg ZoomMsg
    | ViewImgMsg ViewImgMsg
    | ParamsMsg ParamsMsg
    | ParamsInfoMsg ParamsInfoMsg
    | NavigationMsg NavigationMsg
    | PointerMsg PointerMsg


type DragDropMsg
    = DragOver File (List File)
    | Drop File (List File)
    | DragLeave


type ZoomMsg
    = ZoomFit Image
    | ZoomIn
    | ZoomOut
    | ZoomToward ( Float, Float )
    | ZoomAwayFrom ( Float, Float )


type PointerMsg
    = PointerDownRaw Value
      -- = PointerDown ( Float, Float )
    | PointerMove ( Float, Float )
    | PointerUp ( Float, Float )


type ViewImgMsg
    = SelectMovingMode
    | SelectDrawingMode
    | CropCurrentFrame


type ParamsMsg
    = ToggleEqualize Bool
    | ChangeMaxIter String
    | ChangeConvergenceThreshold String
    | ChangeLevels String
    | ChangeSparse String
    | ChangeLambda String
    | ChangeRho String
    | ToggleCrop Bool
    | ChangeCropLeft String
    | ChangeCropTop String
    | ChangeCropRight String
    | ChangeCropBottom String


type ParamsInfoMsg
    = ToggleInfoCrop Bool
    | ToggleInfoMaxIterations Bool
    | ToggleInfoConvergenceThreshold Bool
    | ToggleInfoLevels Bool
    | ToggleInfoSparse Bool
    | ToggleInfoLambda Bool
    | ToggleInfoRho Bool


type NavigationMsg
    = GoToPageImages
    | GoToPageConfig
    | GoToPageRegistration
    | GoToPageLogs


subscriptions : Model -> Sub Msg
subscriptions model =
    case model.state of
        Home _ ->
            Sub.batch [ resizes WindowResizes, imageDecoded ImageDecoded ]

        Loading _ ->
            Sub.batch [ resizes WindowResizes, imageDecoded ImageDecoded ]

        ViewImgs _ ->
            Sub.batch [ resizes WindowResizes, Keyboard.downs KeyDown ]

        Config _ ->
            Sub.batch [ resizes WindowResizes ]

        Registration _ ->
            Sub.batch [ resizes WindowResizes ]

        Logs _ ->
            Sub.batch [ resizes WindowResizes ]


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model.state ) of
        ( NoMsg, _ ) ->
            ( model, Cmd.none )

        ( WindowResizes size, _ ) ->
            let
                viewer =
                    model.viewer
            in
            ( { model
                | device = Device.classify size
                , viewer = { viewer | size = ( size.width, size.height - toFloat headerHeight ) }
              }
            , Cmd.none
            )

        ( DragDropMsg (DragOver _ _), Home _ ) ->
            ( { model | state = Home DraggingSomeFiles }, Cmd.none )

        ( DragDropMsg (Drop file otherFiles), Home _ ) ->
            let
                imageFiles =
                    List.filter (\f -> String.startsWith "image" f.mime) (file :: otherFiles)

                names =
                    Set.fromList (List.map .name imageFiles)
            in
            ( { model | state = Loading { names = names, loaded = Dict.empty } }
            , decodeImages (List.map File.encode imageFiles)
            )

        ( DragDropMsg DragLeave, Home _ ) ->
            ( { model | state = Home Idle }, Cmd.none )

        ( ImageDecoded img, Loading { names, loaded } ) ->
            let
                updatedLoadingState =
                    { names = names
                    , loaded = Dict.insert img.id img loaded
                    }
            in
            if Set.size names == Dict.size updatedLoadingState.loaded then
                case Dict.values updatedLoadingState.loaded of
                    [] ->
                        -- This should be impossible, there must be at least 1 image
                        ( { model | state = Home Idle }, Cmd.none )

                    firstImage :: otherImages ->
                        ( { model
                            | state = ViewImgs { images = Pivot.fromCons firstImage otherImages }
                            , viewer = Viewer.fitImage 1.0 ( toFloat firstImage.width, toFloat firstImage.height ) model.viewer
                          }
                        , Cmd.none
                        )

            else
                ( { model | state = Loading updatedLoadingState }, Cmd.none )

        ( KeyDown rawKey, ViewImgs { images } ) ->
            case Keyboard.navigationKey rawKey of
                Just Keyboard.ArrowRight ->
                    ( { model | state = ViewImgs { images = Pivot.goR images |> Maybe.withDefault (Pivot.goToStart images) } }
                    , Cmd.none
                    )

                Just Keyboard.ArrowLeft ->
                    ( { model | state = ViewImgs { images = Pivot.goL images |> Maybe.withDefault (Pivot.goToEnd images) } }
                    , Cmd.none
                    )

                _ ->
                    ( model, Cmd.none )

        ( ParamsMsg paramsMsg, Config _ ) ->
            ( updateParams paramsMsg model, Cmd.none )

        ( ParamsInfoMsg paramsInfoMsg, Config _ ) ->
            ( { model | paramsInfo = updateParamsInfo paramsInfoMsg model.paramsInfo }, Cmd.none )

        ( NavigationMsg navMsg, ViewImgs data ) ->
            ( goTo navMsg model data, Cmd.none )

        ( NavigationMsg navMsg, Config data ) ->
            ( goTo navMsg model data, Cmd.none )

        ( NavigationMsg navMsg, Registration data ) ->
            ( goTo navMsg model data, Cmd.none )

        ( NavigationMsg navMsg, Logs data ) ->
            ( goTo navMsg model data, Cmd.none )

        ( ZoomMsg zoomMsg, ViewImgs _ ) ->
            ( { model | viewer = zoomViewer zoomMsg model.viewer }, Cmd.none )

        ( PointerMsg pointerMsg, ViewImgs { images } ) ->
            case ( pointerMsg, model.pointerMode ) of
                -- Moving the viewer
                ( PointerDownRaw event, WaitingMove ) ->
                    case Json.Decode.decodeValue Pointer.eventDecoder event of
                        Err _ ->
                            ( model, Cmd.none )

                        Ok { pointer } ->
                            ( { model | pointerMode = PointerMovingFromClientCoords pointer.clientPos }, capture event )

                ( PointerMove ( newX, newY ), PointerMovingFromClientCoords ( x, y ) ) ->
                    ( { model
                        | viewer = Viewer.pan ( newX - x, newY - y ) model.viewer
                        , pointerMode = PointerMovingFromClientCoords ( newX, newY )
                      }
                    , Cmd.none
                    )

                ( PointerUp _, PointerMovingFromClientCoords _ ) ->
                    ( { model | pointerMode = WaitingMove }, Cmd.none )

                -- Drawing the cropped area
                ( PointerDownRaw event, WaitingDraw ) ->
                    case Json.Decode.decodeValue Pointer.eventDecoder event of
                        Err _ ->
                            ( model, Cmd.none )

                        Ok { pointer } ->
                            let
                                ( x, y ) =
                                    Viewer.coordinatesAt pointer.offsetPos model.viewer
                            in
                            ( { model
                                | pointerMode = PointerDrawFromOffsetAndClient pointer.offsetPos pointer.clientPos
                                , bboxDrawn = Just { left = x, top = y, right = x, bottom = y }
                              }
                            , capture event
                            )

                ( PointerMove ( newX, newY ), PointerDrawFromOffsetAndClient ( oX, oY ) ( cX, cY ) ) ->
                    let
                        ( x1, y1 ) =
                            Viewer.coordinatesAt ( oX, oY ) model.viewer

                        ( x2, y2 ) =
                            Viewer.coordinatesAt ( oX + newX - cX, oY + newY - cY ) model.viewer

                        left =
                            min x1 x2

                        top =
                            min y1 y2

                        right =
                            max x1 x2

                        bottom =
                            max y1 y2
                    in
                    ( { model | bboxDrawn = Just { left = left, top = top, right = right, bottom = bottom } }
                    , Cmd.none
                    )

                ( PointerUp _, PointerDrawFromOffsetAndClient _ _ ) ->
                    case model.bboxDrawn of
                        Just { left, right, top, bottom } ->
                            let
                                img =
                                    Pivot.getC images

                                oldParams =
                                    model.params

                                oldParamsForm =
                                    model.paramsForm
                            in
                            if
                                -- sufficient width
                                ((right - left) / model.viewer.scale > 10)
                                    -- sufficient height
                                    && ((bottom - top) / model.viewer.scale > 10)
                                    -- at least one corner inside the image
                                    && (right > 0)
                                    && (left < toFloat img.width)
                                    && (bottom > 0)
                                    && (top < toFloat img.height)
                            then
                                let
                                    leftCrop =
                                        max 0 left

                                    topCrop =
                                        max 0 top

                                    rightCrop =
                                        min (toFloat img.width) right

                                    bottomCrop =
                                        min (toFloat img.height) bottom

                                    newCropParams =
                                        { left = round leftCrop
                                        , top = round topCrop
                                        , right = round rightCrop
                                        , bottom = round bottomCrop
                                        }

                                    newCropForm =
                                        { active = True
                                        , left = NumberInput.updateInt (String.fromInt newCropParams.left) model.paramsForm.crop.left
                                        , top = NumberInput.updateInt (String.fromInt newCropParams.top) model.paramsForm.crop.top
                                        , right = NumberInput.updateInt (String.fromInt newCropParams.right) model.paramsForm.crop.right
                                        , bottom = NumberInput.updateInt (String.fromInt newCropParams.bottom) model.paramsForm.crop.bottom
                                        }
                                in
                                ( { model
                                    | pointerMode = WaitingDraw
                                    , bboxDrawn =
                                        Just
                                            { left = leftCrop
                                            , top = topCrop
                                            , right = rightCrop
                                            , bottom = bottomCrop
                                            }
                                    , params = { oldParams | crop = Just newCropParams }
                                    , paramsForm = { oldParamsForm | crop = newCropForm }
                                  }
                                , Cmd.none
                                )

                            else
                                let
                                    cropForm =
                                        oldParamsForm.crop
                                in
                                ( { model
                                    | pointerMode = WaitingDraw
                                    , bboxDrawn = Nothing
                                    , params = { oldParams | crop = Nothing }
                                    , paramsForm = { oldParamsForm | crop = { cropForm | active = False } }
                                  }
                                , Cmd.none
                                )

                        Nothing ->
                            ( model, Cmd.none )

                _ ->
                    ( model, Cmd.none )

        ( ViewImgMsg SelectMovingMode, ViewImgs _ ) ->
            ( { model | pointerMode = WaitingMove }, Cmd.none )

        ( ViewImgMsg SelectDrawingMode, ViewImgs _ ) ->
            ( { model | pointerMode = WaitingDraw }, Cmd.none )

        ( ViewImgMsg CropCurrentFrame, ViewImgs { images } ) ->
            let
                img =
                    Pivot.getC images

                ( left, top ) =
                    model.viewer.origin

                ( width, height ) =
                    model.viewer.size

                right =
                    left + model.viewer.scale * width

                bottom =
                    top + model.viewer.scale * height

                oldParams =
                    model.params

                oldParamsForm =
                    model.paramsForm
            in
            if
                -- at least one corner inside the image
                (right > 0)
                    && (left < toFloat img.width)
                    && (bottom > 0)
                    && (top < toFloat img.height)
            then
                let
                    leftCrop =
                        max 0 left

                    topCrop =
                        max 0 top

                    rightCrop =
                        min (toFloat img.width) right

                    bottomCrop =
                        min (toFloat img.height) bottom

                    newCropParams =
                        { left = round leftCrop
                        , top = round topCrop
                        , right = round rightCrop
                        , bottom = round bottomCrop
                        }

                    newCropForm =
                        { active = True
                        , left = NumberInput.updateInt (String.fromInt newCropParams.left) model.paramsForm.crop.left
                        , top = NumberInput.updateInt (String.fromInt newCropParams.top) model.paramsForm.crop.top
                        , right = NumberInput.updateInt (String.fromInt newCropParams.right) model.paramsForm.crop.right
                        , bottom = NumberInput.updateInt (String.fromInt newCropParams.bottom) model.paramsForm.crop.bottom
                        }
                in
                ( { model
                    | bboxDrawn =
                        Just
                            { left = leftCrop
                            , top = topCrop
                            , right = rightCrop
                            , bottom = bottomCrop
                            }
                    , params = { oldParams | crop = Just newCropParams }
                    , paramsForm = { oldParamsForm | crop = newCropForm }
                  }
                , Cmd.none
                )

            else
                let
                    cropForm =
                        oldParamsForm.crop
                in
                ( { model
                    | bboxDrawn = Nothing
                    , params = { oldParams | crop = Nothing }
                    , paramsForm = { oldParamsForm | crop = { cropForm | active = False } }
                  }
                , Cmd.none
                )

        _ ->
            ( model, Cmd.none )


zoomViewer : ZoomMsg -> Viewer -> Viewer
zoomViewer msg viewer =
    case msg of
        ZoomFit img ->
            Viewer.fitImage 1.1 ( toFloat img.width, toFloat img.height ) viewer

        ZoomIn ->
            Viewer.zoomIn viewer

        ZoomOut ->
            Viewer.zoomOut viewer

        ZoomToward coordinates ->
            Viewer.zoomToward coordinates viewer

        ZoomAwayFrom coordinates ->
            Viewer.zoomAwayFrom coordinates viewer


goTo : NavigationMsg -> Model -> { images : Pivot Image } -> Model
goTo msg model data =
    case msg of
        GoToPageImages ->
            { model | state = ViewImgs data }

        GoToPageConfig ->
            { model | state = Config data }

        GoToPageRegistration ->
            { model | state = Registration data }

        GoToPageLogs ->
            { model | state = Logs data }


updateParams : ParamsMsg -> Model -> Model
updateParams msg ({ params, paramsForm } as model) =
    case msg of
        ToggleEqualize equalize ->
            { model | params = { params | equalize = equalize } }

        ChangeMaxIter str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.maxIterations

                updatedForm =
                    { paramsForm | maxIterations = updatedField }
            in
            case updatedField.decodedInput of
                Ok maxIterations ->
                    { model
                        | params = { params | maxIterations = maxIterations }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ChangeConvergenceThreshold str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.convergenceThreshold

                updatedForm =
                    { paramsForm | convergenceThreshold = updatedField }
            in
            case updatedField.decodedInput of
                Ok convergenceThreshold ->
                    { model
                        | params = { params | convergenceThreshold = convergenceThreshold }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ChangeLevels str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.levels

                updatedForm =
                    { paramsForm | levels = updatedField }
            in
            case updatedField.decodedInput of
                Ok levels ->
                    { model
                        | params = { params | levels = levels }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ChangeSparse str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.sparse

                updatedForm =
                    { paramsForm | sparse = updatedField }
            in
            case updatedField.decodedInput of
                Ok sparse ->
                    { model
                        | params = { params | sparse = sparse }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ChangeLambda str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.lambda

                updatedForm =
                    { paramsForm | lambda = updatedField }
            in
            case updatedField.decodedInput of
                Ok lambda ->
                    { model
                        | params = { params | lambda = lambda }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ChangeRho str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.rho

                updatedForm =
                    { paramsForm | rho = updatedField }
            in
            case updatedField.decodedInput of
                Ok rho ->
                    { model
                        | params = { params | rho = rho }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

        ToggleCrop activeCrop ->
            let
                newCropForm =
                    CropForm.toggle activeCrop paramsForm.crop
            in
            case ( activeCrop, CropForm.decoded newCropForm ) of
                ( True, Just ({ left, top, right, bottom } as crop) ) ->
                    { model
                        | params = { params | crop = Just crop }
                        , paramsForm = { paramsForm | crop = newCropForm }
                        , bboxDrawn =
                            Just
                                { left = toFloat left
                                , top = toFloat top
                                , right = toFloat right
                                , bottom = toFloat bottom
                                }
                    }

                _ ->
                    { model
                        | params = { params | crop = Nothing }
                        , paramsForm = { paramsForm | crop = newCropForm }
                        , bboxDrawn = Nothing
                    }

        ChangeCropLeft str ->
            let
                newCropForm =
                    CropForm.updateLeft str paramsForm.crop
            in
            { model
                | params = { params | crop = CropForm.decoded newCropForm }
                , paramsForm = { paramsForm | crop = newCropForm }
            }

        ChangeCropTop str ->
            let
                newCropForm =
                    CropForm.updateTop str paramsForm.crop
            in
            { model
                | params = { params | crop = CropForm.decoded newCropForm }
                , paramsForm = { paramsForm | crop = newCropForm }
            }

        ChangeCropRight str ->
            let
                newCropForm =
                    CropForm.updateRight str paramsForm.crop
            in
            { model
                | params = { params | crop = CropForm.decoded newCropForm }
                , paramsForm = { paramsForm | crop = newCropForm }
            }

        ChangeCropBottom str ->
            let
                newCropForm =
                    CropForm.updateBottom str paramsForm.crop
            in
            { model
                | params = { params | crop = CropForm.decoded newCropForm }
                , paramsForm = { paramsForm | crop = newCropForm }
            }


updateParamsInfo : ParamsInfoMsg -> ParametersToggleInfo -> ParametersToggleInfo
updateParamsInfo msg toggleInfo =
    case msg of
        ToggleInfoCrop visible ->
            { toggleInfo | crop = visible }

        ToggleInfoMaxIterations visible ->
            { toggleInfo | maxIterations = visible }

        ToggleInfoConvergenceThreshold visible ->
            { toggleInfo | convergenceThreshold = visible }

        ToggleInfoLevels visible ->
            { toggleInfo | levels = visible }

        ToggleInfoSparse visible ->
            { toggleInfo | sparse = visible }

        ToggleInfoLambda visible ->
            { toggleInfo | lambda = visible }

        ToggleInfoRho visible ->
            { toggleInfo | rho = visible }



-- View ##############################################################


view : Model -> Html Msg
view model =
    Element.layout [ Style.font, Element.clip ]
        (viewElmUI model)


viewElmUI : Model -> Element Msg
viewElmUI model =
    case model.state of
        Home draggingState ->
            viewHome draggingState

        Loading loadData ->
            viewLoading loadData

        ViewImgs { images } ->
            viewImgs model.pointerMode model.bboxDrawn model.viewer images

        Config { images } ->
            viewConfig model.params model.paramsForm model.paramsInfo

        Registration { images } ->
            viewRegistration

        Logs { images } ->
            viewLogs



-- Header


type PageHeader
    = PageImages
    | PageConfig
    | PageRegistration
    | PageLogs


{-| WARNING: this has to be kept consistent with the text size in the header
-}
headerHeight : Int
headerHeight =
    40


headerBar : List ( PageHeader, Bool ) -> Element Msg
headerBar pages =
    Element.row
        [ height (Element.px headerHeight)
        , centerX
        ]
        (List.map (\( page, current ) -> pageHeaderElement current page) pages)


pageHeaderElement : Bool -> PageHeader -> Element Msg
pageHeaderElement current page =
    let
        bgColor =
            if current then
                Style.almostWhite

            else
                Style.white

        attributes =
            [ Element.Background.color bgColor
            , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
            , padding 10
            , height (Element.px headerHeight)
            ]
    in
    case page of
        PageImages ->
            Element.Input.button attributes
                { onPress =
                    if current then
                        Nothing

                    else
                        Just (NavigationMsg GoToPageImages)
                , label = Element.text "Images"
                }

        PageConfig ->
            Element.Input.button attributes
                { onPress =
                    if current then
                        Nothing

                    else
                        Just (NavigationMsg GoToPageConfig)
                , label = Element.text "Config"
                }

        PageRegistration ->
            Element.Input.button attributes
                { onPress =
                    if current then
                        Nothing

                    else
                        Just (NavigationMsg GoToPageRegistration)
                , label = Element.text "Registration"
                }

        PageLogs ->
            Element.Input.button attributes
                { onPress =
                    if current then
                        Nothing

                    else
                        Just (NavigationMsg GoToPageLogs)
                , label = Element.text "Logs"
                }



-- Logs


viewLogs : Element Msg
viewLogs =
    Element.column [ width fill ]
        [ headerBar
            [ ( PageImages, False )
            , ( PageConfig, False )
            , ( PageRegistration, False )
            , ( PageLogs, True )
            ]
        ]



-- Registration


viewRegistration : Element Msg
viewRegistration =
    Element.column [ width fill ]
        [ headerBar
            [ ( PageImages, False )
            , ( PageConfig, False )
            , ( PageRegistration, True )
            , ( PageLogs, False )
            ]
        ]



-- Parameters config


viewConfig : Parameters -> ParametersForm -> ParametersToggleInfo -> Element Msg
viewConfig params paramsForm paramsInfo =
    Element.column [ width fill ]
        [ headerBar
            [ ( PageImages, False )
            , ( PageConfig, True )
            , ( PageRegistration, False )
            , ( PageLogs, False )
            ]
        , Element.column [ paddingXY 20 32, spacing 32, centerX ]
            [ runButton paramsForm

            -- Title
            , Element.el [ Element.Font.center, Element.Font.size 32 ] (Element.text "Algorithm parameters")

            -- Cropped working frame
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text "Cropped working frame:"
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoCrop
                        , icon = infoIcon
                        , checked = paramsInfo.crop
                        , label = Element.Input.labelHidden "Show detail info about cropped working frame"
                        }
                    ]
                , moreInfo paramsInfo.crop "Instead of using the whole image to estimate the registration, it is often faster and as accurate to focus the algorithm attention on a smaller frame in the image. The parameters here are the left, top, right and bottom coordinates of that cropped frame on which we want the algorithm to focus when estimating the alignment parameters."
                , Element.row [ spacing 10 ]
                    [ Element.text "off"
                    , toggle (ParamsMsg << ToggleCrop) paramsForm.crop.active 30 "Toggle cropped working frame"
                    , Element.text "on"
                    ]
                , CropForm.boxEditor
                    { changeLeft = ParamsMsg << ChangeCropLeft
                    , changeTop = ParamsMsg << ChangeCropTop
                    , changeRight = ParamsMsg << ChangeCropRight
                    , changeBottom = ParamsMsg << ChangeCropBottom
                    }
                    paramsForm.crop
                , displayErrors (CropForm.errors paramsForm.crop)
                ]

            -- Equalize mean intensities
            , Element.column [ spacing 10 ]
                [ Element.text "Equalize mean intensities:"
                , Element.row [ spacing 10 ]
                    [ Element.text "off"
                    , toggle (ParamsMsg << ToggleEqualize) params.equalize 30 "Toggle mean intensities equalization"
                    , Element.text "on"
                    ]
                ]

            -- Maximum number of iterations
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text "Maximum number of iterations:"
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoMaxIterations
                        , icon = infoIcon
                        , checked = paramsInfo.maxIterations
                        , label = Element.Input.labelHidden "Show detail info about the maximum number of iterations"
                        }
                    ]
                , moreInfo paramsInfo.maxIterations "This is the maximum number of iterations allowed per level. If this is reached, the algorithm stops whether it converged or not."
                , Element.text ("(default to " ++ String.fromInt defaultParams.maxIterations ++ ")")
                , intInput paramsForm.maxIterations (ParamsMsg << ChangeMaxIter) "Maximum number of iterations"
                , displayIntErrors paramsForm.maxIterations.decodedInput
                ]

            -- Convergence threshold
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text "Convergence threshold:"
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoConvergenceThreshold
                        , icon = infoIcon
                        , checked = paramsInfo.convergenceThreshold
                        , label = Element.Input.labelHidden "Show detail info about the convergence threshold parameter"
                        }
                    ]
                , moreInfo paramsInfo.convergenceThreshold "The algorithm stops when the relative error difference between to estimates falls below this value."
                , Element.text ("(default to " ++ String.fromFloat defaultParams.convergenceThreshold ++ ")")
                , floatInput paramsForm.convergenceThreshold (ParamsMsg << ChangeConvergenceThreshold) "Convergence threshold"
                , displayFloatErrors paramsForm.convergenceThreshold.decodedInput
                ]

            -- Multi-resolution pyramid levels
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text "Number of pyramid levels:"
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoLevels
                        , icon = infoIcon
                        , checked = paramsInfo.levels
                        , label = Element.Input.labelHidden "Show detail info about the levels parameter"
                        }
                    ]
                , moreInfo paramsInfo.levels "The number of levels for the multi-resolution approach. Each level halves/doubles the resolution of the previous one. The algorithm starts at the lowest resolution and transfers the converged parameters at one resolution to the initialization of the next. Increasing the number of levels enables better convergence for bigger movements but too many levels might make it definitively drift away. Targetting a lowest resolution of about 100x100 is generally good enough. The number of levels also has a joint interaction with the sparse threshold parameter so keep that in mind while changing this parameter."
                , Element.text ("(default to " ++ String.fromInt defaultParams.levels ++ ")")
                , intInput paramsForm.levels (ParamsMsg << ChangeLevels) "Number of pyramid levels"
                , displayIntErrors paramsForm.levels.decodedInput
                ]

            -- Sparse ratio threshold
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text "Sparse ratio threshold to switch:"
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoSparse
                        , icon = infoIcon
                        , checked = paramsInfo.sparse
                        , label = Element.Input.labelHidden "Show detail info about the sparse parameter"
                        }
                    ]
                , moreInfo paramsInfo.sparse "Sparse ratio threshold to switch between dense and sparse registration. At each pyramid level only the pixels with the highest gradient intensities are kept, making each level sparser than the previous one. Once the ratio of selected pixels goes below this sparse ratio parameter, the algorithm performs a sparse registration, using only the selected points at that level. If you want to use a dense registration at every level, you can set this parameter to 0."
                , Element.text ("(default to " ++ String.fromFloat defaultParams.sparse ++ ")")
                , floatInput paramsForm.sparse (ParamsMsg << ChangeSparse) "Sparse ratio threshold to switch"
                , displayFloatErrors paramsForm.sparse.decodedInput
                ]

            -- lambda
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text ("lambda: (default to " ++ String.fromFloat defaultParams.lambda ++ ")")
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoLambda
                        , icon = infoIcon
                        , checked = paramsInfo.lambda
                        , label = Element.Input.labelHidden "Show detail info about the lambda parameter"
                        }
                    ]
                , moreInfo paramsInfo.lambda "Weight of the L1 term (high means no correction)."
                , floatInput paramsForm.lambda (ParamsMsg << ChangeLambda) "lambda"
                , displayFloatErrors paramsForm.lambda.decodedInput
                ]

            -- rho
            , Element.column [ spacing 10 ]
                [ Element.row [ spacing 10 ]
                    [ Element.text ("rho: (default to " ++ String.fromFloat defaultParams.rho ++ ")")
                    , Element.Input.checkbox []
                        { onChange = ParamsInfoMsg << ToggleInfoRho
                        , icon = infoIcon
                        , checked = paramsInfo.rho
                        , label = Element.Input.labelHidden "Show detail info about the rho parameter"
                        }
                    ]
                , moreInfo paramsInfo.rho "Lagrangian penalty."
                , floatInput paramsForm.rho (ParamsMsg << ChangeRho) "rho"
                , displayFloatErrors paramsForm.rho.decodedInput
                ]
            ]
        ]



-- More info


moreInfo : Bool -> String -> Element msg
moreInfo visible message =
    if not visible then
        Element.none

    else
        Element.paragraph
            [ Element.Background.color Style.almostWhite
            , padding 10
            , Element.Font.size 14
            , width (Element.maximum 400 fill)
            ]
            [ Element.text message ]


infoIcon : Bool -> Element msg
infoIcon detailsVisible =
    if detailsVisible then
        Element.el
            [ Element.Border.width 1
            , Element.Border.rounded 4
            , Element.Font.center
            , width (Element.px 24)
            , height (Element.px 24)
            , Element.Border.solid
            , Element.Background.color Style.almostWhite
            ]
            (Element.text "?")

    else
        Element.el
            [ Element.Border.width 1
            , Element.Border.rounded 4
            , Element.Font.center
            , width (Element.px 24)
            , height (Element.px 24)
            , Element.Border.dashed
            ]
            (Element.text "?")



-- Run button


runButton : ParametersForm -> Element Msg
runButton paramsForm =
    let
        hasNoError =
            List.isEmpty (CropForm.errors paramsForm.crop)
                && isOk paramsForm.maxIterations.decodedInput
                && isOk paramsForm.convergenceThreshold.decodedInput
                && isOk paramsForm.levels.decodedInput
                && isOk paramsForm.sparse.decodedInput
                && isOk paramsForm.lambda.decodedInput
                && isOk paramsForm.rho.decodedInput
    in
    if hasNoError then
        Element.Input.button
            [ centerX
            , padding 12
            , Element.Border.solid
            , Element.Border.width 1
            , Element.Border.rounded 4
            ]
            { onPress = Nothing, label = Element.text "Run ▶" }

    else
        Element.Input.button
            [ centerX
            , padding 12
            , Element.Border.solid
            , Element.Border.width 1
            , Element.Border.rounded 4
            , Element.Font.color Style.lightGrey
            ]
            { onPress = Nothing, label = Element.text "Run ▶" }


isOk : Result err ok -> Bool
isOk result =
    case result of
        Err _ ->
            False

        Ok _ ->
            True



-- Crop input


displayErrors : List String -> Element msg
displayErrors errors =
    if List.isEmpty errors then
        Element.none

    else
        Element.column [ spacing 10, Element.Font.size 14, Element.Font.color Style.errorColor ]
            (List.map (\err -> Element.paragraph [] [ Element.text err ]) errors)



-- Int input


displayIntErrors : Result (List NumberInput.IntError) a -> Element msg
displayIntErrors result =
    case result of
        Ok _ ->
            Element.none

        Err errors ->
            displayErrors (List.map (NumberInput.intErrorToString { valueName = "Value" }) errors)


intInput : NumberInput.Field Int NumberInput.IntError -> (String -> msg) -> String -> Element msg
intInput field msgTag label =
    let
        textField =
            Element.Input.text [ Element.Border.width 0, Element.Font.center, width (Element.px 100) ]
                { onChange = msgTag
                , text = field.input
                , placeholder = Nothing
                , label = Element.Input.labelHidden label
                }
    in
    case field.decodedInput of
        Err _ ->
            Element.row
                [ Element.Border.solid
                , Element.Border.width 1
                , Element.Border.rounded 4
                , Element.Font.color Style.errorColor
                ]
                [ numberSideButton Nothing "−"
                , textField
                , numberSideButton Nothing "+"
                ]

        Ok current ->
            let
                increased =
                    field.increase current

                decreased =
                    field.decrease current

                decrementMsg =
                    case field.min of
                        Nothing ->
                            Just (msgTag (String.fromInt decreased))

                        Just minBound ->
                            if current <= minBound then
                                Nothing

                            else
                                Just (msgTag (String.fromInt <| max decreased minBound))

                incrementMsg =
                    case field.max of
                        Nothing ->
                            Just (msgTag (String.fromInt increased))

                        Just maxBound ->
                            if current >= maxBound then
                                Nothing

                            else
                                Just (msgTag (String.fromInt <| min increased maxBound))
            in
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton decrementMsg "−"
                , textField
                , numberSideButton incrementMsg "+"
                ]


numberSideButton : Maybe msg -> String -> Element msg
numberSideButton maybeMsg label =
    let
        textColor =
            if maybeMsg == Nothing then
                Style.lightGrey

            else
                Style.black
    in
    Element.Input.button
        [ height fill
        , width (Element.px 44)
        , Element.Font.center
        , Element.Font.color textColor
        , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
        ]
        { onPress = maybeMsg, label = Element.text label }



-- Float input


displayFloatErrors : Result (List NumberInput.FloatError) a -> Element msg
displayFloatErrors result =
    case result of
        Ok _ ->
            Element.none

        Err errors ->
            displayErrors (List.map (NumberInput.floatErrorToString { valueName = "Value" }) errors)


floatInput : NumberInput.Field Float NumberInput.FloatError -> (String -> msg) -> String -> Element msg
floatInput field msgTag label =
    let
        textField =
            Element.Input.text [ Element.Border.width 0, Element.Font.center, width (Element.px 140) ]
                { onChange = msgTag
                , text = field.input
                , placeholder = Nothing
                , label = Element.Input.labelHidden label
                }
    in
    case field.decodedInput of
        Err _ ->
            Element.row
                [ Element.Border.solid
                , Element.Border.width 1
                , Element.Border.rounded 4
                , Element.Font.color Style.errorColor
                ]
                [ numberSideButton Nothing "−"
                , textField
                , numberSideButton Nothing "+"
                ]

        Ok current ->
            let
                increased =
                    field.increase current

                decreased =
                    field.decrease current

                decrementMsg =
                    case field.min of
                        Nothing ->
                            Just (msgTag (String.fromFloat decreased))

                        Just minBound ->
                            if current <= minBound then
                                Nothing

                            else
                                Just (msgTag (String.fromFloat <| max decreased minBound))

                incrementMsg =
                    case field.max of
                        Nothing ->
                            Just (msgTag (String.fromFloat increased))

                        Just maxBound ->
                            if current >= maxBound then
                                Nothing

                            else
                                Just (msgTag (String.fromFloat <| min increased maxBound))
            in
            Element.row [ Element.Border.solid, Element.Border.width 1, Element.Border.rounded 4 ]
                [ numberSideButton decrementMsg "−"
                , textField
                , numberSideButton incrementMsg "+"
                ]



-- toggle


toggle : (Bool -> Msg) -> Bool -> Float -> String -> Element Msg
toggle msg checked toggleHeight label =
    Element.Input.checkbox [] <|
        { onChange = msg
        , label = Element.Input.labelHidden label
        , checked = checked
        , icon =
            toggleCheckboxWidget
                { offColor = Style.lightGrey
                , onColor = Style.green
                , sliderColor = Style.white
                , toggleWidth = 2 * round toggleHeight
                , toggleHeight = round toggleHeight
                }
        }


toggleCheckboxWidget : { offColor : Element.Color, onColor : Element.Color, sliderColor : Element.Color, toggleWidth : Int, toggleHeight : Int } -> Bool -> Element msg
toggleCheckboxWidget { offColor, onColor, sliderColor, toggleWidth, toggleHeight } checked =
    let
        pad =
            3

        sliderSize =
            toggleHeight - 2 * pad

        translation =
            (toggleWidth - sliderSize - pad)
                |> String.fromInt
    in
    Element.el
        [ Element.Background.color <|
            if checked then
                onColor

            else
                offColor
        , Element.width <| Element.px <| toggleWidth
        , Element.height <| Element.px <| toggleHeight
        , Element.Border.rounded (toggleHeight // 2)
        , Element.inFront <|
            Element.el [ Element.height Element.fill ] <|
                Element.el
                    [ Element.Background.color sliderColor
                    , Element.Border.rounded <| sliderSize // 2
                    , Element.width <| Element.px <| sliderSize
                    , Element.height <| Element.px <| sliderSize
                    , Element.centerY
                    , Element.moveRight pad
                    , Element.htmlAttribute <|
                        Html.Attributes.style "transition" ".4s"
                    , Element.htmlAttribute <|
                        if checked then
                            Html.Attributes.style "transform" <| "translateX(" ++ translation ++ "px)"

                        else
                            Html.Attributes.class ""
                    ]
                    Element.none
        ]
        Element.none



-- View Images


viewImgs : PointerMode -> Maybe BBox -> Viewer -> Pivot Image -> Element Msg
viewImgs pointerMode bboxDrawn viewer images =
    let
        img =
            Pivot.getC images

        clickButton abled msg title icon =
            let
                strokeColor =
                    if abled then
                        Style.black

                    else
                        Style.lightGrey
            in
            Element.Input.button
                [ padding 6
                , centerX
                , Element.Background.color (Element.rgba255 255 255 255 0.8)
                , Element.Font.color strokeColor
                , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
                , Element.htmlAttribute <| Html.Attributes.title title
                ]
                { onPress = Just msg
                , label = icon 32
                }

        modeButton selected msg title icon =
            let
                ( bgColor, action ) =
                    if selected then
                        ( Style.lightGrey, Nothing )

                    else
                        ( Element.rgba 255 255 255 0.8, Just msg )
            in
            Element.Input.button
                [ padding 6
                , centerX
                , Element.Background.color bgColor
                , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
                , Element.htmlAttribute <| Html.Attributes.title title
                ]
                { onPress = action
                , label = icon 32
                }

        isMovingMode =
            case pointerMode of
                WaitingMove ->
                    True

                PointerMovingFromClientCoords _ ->
                    True

                WaitingDraw ->
                    False

                PointerDrawFromOffsetAndClient _ _ ->
                    False

        buttonsRow =
            Element.row [ width fill ]
                [ clickButton True (ZoomMsg (ZoomFit img)) "Fit zoom to image" Icon.zoomFit
                , clickButton True (ZoomMsg ZoomOut) "Zoom out" Icon.zoomOut
                , clickButton True (ZoomMsg ZoomIn) "Zoom in" Icon.zoomIn
                , modeButton isMovingMode (ViewImgMsg SelectMovingMode) "Move mode" Icon.move
                , Element.el [ width (Element.maximum 100 fill) ] Element.none
                , modeButton (not isMovingMode) (ViewImgMsg SelectDrawingMode) "Draw the cropped working area as a bounding box" Icon.boundingBox
                , clickButton True (ViewImgMsg CropCurrentFrame) "Set the cropped working area to the current frame" Icon.maximize
                ]

        imgSvgAttributes =
            [ Svg.Attributes.xlinkHref img.url
            , Svg.Attributes.width (String.fromInt img.width)
            , Svg.Attributes.height (String.fromInt img.height)
            , Svg.Attributes.class "pixelated"
            ]

        ( viewerWidth, viewerHeight ) =
            viewer.size

        viewerAttributes =
            [ Html.Attributes.style "pointer-events" "none"
            , Viewer.Svg.transform viewer
            ]

        svgContent =
            case bboxDrawn of
                Nothing ->
                    Svg.g viewerAttributes [ Svg.image imgSvgAttributes [] ]

                Just { left, top, right, bottom } ->
                    let
                        bboxWidth =
                            right - left

                        bboxHeight =
                            bottom - top

                        strokeWidth =
                            viewer.scale * 2
                    in
                    Svg.g viewerAttributes
                        [ Svg.image imgSvgAttributes []
                        , Svg.rect
                            [ Svg.Attributes.x (String.fromFloat left)
                            , Svg.Attributes.y (String.fromFloat top)
                            , Svg.Attributes.width (String.fromFloat bboxWidth)
                            , Svg.Attributes.height (String.fromFloat bboxHeight)
                            , Svg.Attributes.fill "white"
                            , Svg.Attributes.fillOpacity "0.3"
                            , Svg.Attributes.stroke "red"
                            , Svg.Attributes.strokeWidth (String.fromFloat strokeWidth)
                            ]
                            []
                        ]

        svgViewer =
            Element.html <|
                Svg.svg
                    [ Html.Attributes.width (floor viewerWidth)
                    , Html.Attributes.height (floor viewerHeight)
                    , Wheel.onWheel (zoomWheelMsg viewer)
                    , msgOn "pointerdown" (Json.Decode.map (PointerMsg << PointerDownRaw) Json.Decode.value)
                    , Pointer.onUp (\e -> PointerMsg (PointerUp e.pointer.offsetPos))
                    , Html.Attributes.style "touch-action" "none"
                    , Html.Events.preventDefaultOn "pointermove" <|
                        Json.Decode.map (\coords -> ( PointerMsg (PointerMove coords), True )) <|
                            Json.Decode.map2 Tuple.pair
                                (Json.Decode.field "clientX" Json.Decode.float)
                                (Json.Decode.field "clientY" Json.Decode.float)
                    ]
                    [ svgContent ]
    in
    Element.column [ height fill ]
        [ headerBar
            [ ( PageImages, True )
            , ( PageConfig, False )
            , ( PageRegistration, False )
            , ( PageLogs, False )
            ]
        , Element.html <|
            Html.node "style"
                []
                [ Html.text ".pixelated { image-rendering: pixelated; image-rendering: crisp-edges; }" ]
        , Element.el
            [ Element.inFront buttonsRow
            , Element.clip
            , height fill
            ]
            svgViewer
        ]


msgOn : String -> Decoder msg -> Html.Attribute msg
msgOn event =
    Json.Decode.map (\msg -> { message = msg, stopPropagation = True, preventDefault = True })
        >> Html.Events.custom event


zoomWheelMsg : Viewer -> Wheel.Event -> Msg
zoomWheelMsg viewer event =
    let
        coordinates =
            Viewer.coordinatesAt event.mouseEvent.offsetPos viewer
    in
    if event.deltaY > 0 then
        ZoomMsg (ZoomAwayFrom coordinates)

    else
        ZoomMsg (ZoomToward coordinates)


viewHome : FileDraggingState -> Element Msg
viewHome draggingState =
    Element.column (padding 20 :: width fill :: height fill :: onDropAttributes)
        [ viewTitle
        , dropAndLoadArea draggingState
        ]


viewLoading : { names : Set String, loaded : Dict String Image } -> Element Msg
viewLoading { names, loaded } =
    let
        totalCount =
            Set.size names

        loadCount =
            Dict.size loaded
    in
    Element.column [ padding 20, width fill, height fill ]
        [ viewTitle
        , Element.el [ width fill, height fill ]
            (Element.column
                [ centerX, centerY, spacing 32 ]
                [ Element.el loadingBoxBorderAttributes (loadBar loadCount totalCount)
                , Element.el [ centerX ] (Element.text ("Loading " ++ String.fromInt totalCount ++ " images"))
                ]
            )
        ]


loadBar : Int -> Int -> Element msg
loadBar loaded total =
    let
        barLength =
            (400 - 2 * 4) * loaded // total
    in
    Element.el
        [ width (Element.px barLength)
        , height Element.fill
        , Element.Background.color Style.dropColor
        , Element.htmlAttribute
            (Transition.properties
                [ Transition.property "width" 200 [] ]
            )
        ]
        Element.none


viewTitle : Element msg
viewTitle =
    Element.column [ centerX, spacing 16 ]
        [ Element.paragraph [ Element.Font.center, Element.Font.size 32 ] [ Element.text "Low rank image registration" ]
        , Element.row [ alignRight, spacing 8 ]
            [ Element.link [ Element.Font.underline ]
                { url = "https://github.com/mpizenberg/lowrr", label = Element.text "code on GitHub" }
            , Element.el [] Element.none
            , Icon.github 16
            ]
        , Element.row [ alignRight, spacing 8 ]
            [ Element.link [ Element.Font.underline ]
                { url = "https://hal.archives-ouvertes.fr/hal-03172399", label = Element.text "read the paper" }
            , Element.el [] Element.none
            , Icon.fileText 16
            ]
        ]


dropAndLoadArea : FileDraggingState -> Element Msg
dropAndLoadArea draggingState =
    let
        borderStyle =
            case draggingState of
                Idle ->
                    Element.Border.dashed

                DraggingSomeFiles ->
                    Element.Border.solid

        dropOrLoadText =
            Element.row []
                [ Element.text "Drop images or "
                , Element.html
                    (File.hiddenInputMultiple
                        "TheFileInput"
                        [ "image/*" ]
                        (\file otherFiles -> DragDropMsg (Drop file otherFiles))
                    )
                , Element.el [ Element.Font.underline ]
                    (Element.html
                        (Html.label [ Html.Attributes.for "TheFileInput", Html.Attributes.style "cursor" "pointer" ]
                            [ Html.text "load from disk" ]
                        )
                    )
                ]
    in
    Element.el [ width fill, height fill ]
        (Element.column [ centerX, centerY, spacing 32 ]
            [ Element.el (dropIconBorderAttributes borderStyle) (Icon.arrowDown 48)
            , dropOrLoadText
            ]
        )


dropIconBorderAttributes : Element.Attribute msg -> List (Element.Attribute msg)
dropIconBorderAttributes dashedAttribute =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , centerX
    , Element.clip

    -- below is different
    , paddingXY 16 16
    , dashedAttribute
    , Element.Border.rounded 16
    , height (Element.px (48 + (16 + 4) * 2))
    , width (Element.px (48 + (16 + 4) * 2))
    , borderTransition
    ]


loadingBoxBorderAttributes : List (Element.Attribute msg)
loadingBoxBorderAttributes =
    [ Element.Border.width 4
    , Element.Font.color Style.dropColor
    , centerX
    , Element.clip

    -- below is different
    , paddingXY 0 0
    , Element.Border.solid
    , Element.Border.rounded 0
    , height (Element.px ((16 + 4) * 2))
    , width (Element.px 325)
    , borderTransition
    ]


borderTransition : Element.Attribute msg
borderTransition =
    Element.htmlAttribute
        (Transition.properties
            [ Transition.property "border-radius" 300 []
            , Transition.property "height" 300 []
            , Transition.property "width" 300 []
            ]
        )


onDropAttributes : List (Element.Attribute Msg)
onDropAttributes =
    List.map Element.htmlAttribute
        (File.onDrop
            { onOver = \file otherFiles -> DragDropMsg (DragOver file otherFiles)
            , onDrop = \file otherFiles -> DragDropMsg (Drop file otherFiles)
            , onLeave = Just { id = "FileDropArea", msg = DragDropMsg DragLeave }
            }
        )
