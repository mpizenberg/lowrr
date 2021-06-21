port module Main exposing (main)

import Browser
import Browser.Dom
import Browser.Navigation
import Canvas
import Canvas.Settings
import Canvas.Settings.Advanced
import Canvas.Settings.Line
import Canvas.Texture exposing (Texture)
import Color
import CropForm
import Device exposing (Device)
import Dict exposing (Dict)
import Element exposing (Element, alignBottom, alignLeft, alignRight, centerX, centerY, fill, height, padding, paddingXY, spacing, width)
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
import Json.Encode exposing (Value)
import Keyboard exposing (RawKey)
import NumberInput
import Pivot exposing (Pivot)
import Process
import Set exposing (Set)
import Simple.Transition as Transition
import Style
import Svg
import Svg.Attributes
import Task
import Viewer exposing (Viewer)
import Viewer.Canvas


port resizes : (Device.Size -> msg) -> Sub msg


port decodeImages : List Value -> Cmd msg


port loadImagesFromUrls : List String -> Cmd msg


port imageDecoded : ({ id : String, img : Value } -> msg) -> Sub msg


port capture : Value -> Cmd msg


port run : Value -> Cmd msg


port stop : () -> Cmd msg


port saveRegisteredImages : Int -> Cmd msg


port log : ({ lvl : Int, content : String } -> msg) -> Sub msg


port updateRunStep : ({ step : String, progress : Maybe Int } -> msg) -> Sub msg


port receiveCroppedImages : (List { id : String, img : Value } -> msg) -> Sub msg


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
    , registeredViewer : Viewer
    , pointerMode : PointerMode
    , bboxDrawn : Maybe BBox
    , registeredImages : Maybe (Pivot Image)
    , seenLogs : List { lvl : Int, content : String }
    , notSeenLogs : List { lvl : Int, content : String }
    , scrollPos : Float
    , verbosity : Int
    , autoscroll : Bool
    , runStep : RunStep
    , imagesCount : Int
    }


type RunStep
    = StepNotStarted
    | StepMultiresPyramid
    | StepLevel Int
    | StepIteration Int Int
    | StepApplying Int
    | StepEncoding Int
    | StepDone
    | StepSaving Int


type alias BBox =
    { left : Float
    , top : Float
    , right : Float
    , bottom : Float
    }


type State
    = Home FileDraggingState
    | Loading { names : Set String, loaded : Dict String Image }
    | LoadingError
    | ViewImgs { images : Pivot Image }
    | Config { images : Pivot Image }
    | Registration { images : Pivot Image }
    | Logs { images : Pivot Image }


type FileDraggingState
    = Idle
    | DraggingSomeFiles


type alias Image =
    { id : String
    , texture : Texture
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
    , maxVerbosity : Int
    }


encodeParams : Parameters -> Value
encodeParams params =
    Json.Encode.object
        [ ( "crop", encodeMaybe encodeCrop params.crop )
        , ( "equalize", Json.Encode.bool params.equalize )
        , ( "levels", Json.Encode.int params.levels )
        , ( "sparse", Json.Encode.float params.sparse )
        , ( "lambda", Json.Encode.float params.lambda )
        , ( "rho", Json.Encode.float params.rho )
        , ( "maxIterations", Json.Encode.int params.maxIterations )
        , ( "convergenceThreshold", Json.Encode.float params.convergenceThreshold )
        , ( "maxVerbosity", Json.Encode.int params.maxVerbosity )
        ]


encodeMaybe : (a -> Value) -> Maybe a -> Value
encodeMaybe encoder data =
    Maybe.withDefault Json.Encode.null (Maybe.map encoder data)


type alias Crop =
    { left : Int
    , top : Int
    , right : Int
    , bottom : Int
    }


encodeCrop : Crop -> Value
encodeCrop { left, top, right, bottom } =
    Json.Encode.object
        [ ( "left", Json.Encode.int left )
        , ( "top", Json.Encode.int top )
        , ( "right", Json.Encode.int right )
        , ( "bottom", Json.Encode.int bottom )
        ]


type alias ParametersForm =
    { crop : CropForm.State
    , maxIterations : NumberInput.Field Int NumberInput.IntError
    , convergenceThreshold : NumberInput.Field Float NumberInput.FloatError
    , levels : NumberInput.Field Int NumberInput.IntError
    , sparse : NumberInput.Field Float NumberInput.FloatError
    , lambda : NumberInput.Field Float NumberInput.FloatError
    , rho : NumberInput.Field Float NumberInput.FloatError
    , maxVerbosity : NumberInput.Field Int NumberInput.IntError
    }


type alias ParametersToggleInfo =
    { crop : Bool
    , maxIterations : Bool
    , convergenceThreshold : Bool
    , levels : Bool
    , sparse : Bool
    , lambda : Bool
    , rho : Bool
    , maxVerbosity : Bool
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
    -- initialModel size
    --     |> (\m -> { m | state = Loading { names = Set.singleton "img", loaded = Dict.empty } })
    --     |> update (ImageDecoded { id = "img", url = "/img/pano_bayeux.jpg", width = 2000, height = 225 })
    ( initialModel size, Cmd.none )


initialModel : Device.Size -> Model
initialModel size =
    { state = Home Idle
    , device = Device.classify size
    , params = defaultParams
    , paramsForm = defaultParamsForm
    , paramsInfo = defaultParamsInfo
    , viewer = Viewer.withSize ( size.width, size.height - toFloat (headerHeight + progressBarHeight) )
    , registeredViewer = Viewer.withSize ( size.width, size.height - toFloat (headerHeight + progressBarHeight) )
    , pointerMode = WaitingMove
    , bboxDrawn = Nothing
    , registeredImages = Nothing
    , seenLogs = []
    , notSeenLogs = []
    , scrollPos = 0.0
    , verbosity = 2
    , autoscroll = True
    , runStep = StepNotStarted
    , imagesCount = 0
    }


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
    , maxVerbosity = 3
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
    , maxVerbosity =
        { anyInt | min = Just 0, max = Just 4 }
            |> NumberInput.setDefaultIntValue defaultParams.maxVerbosity
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
    , maxVerbosity = False
    }



-- Update ############################################################


type Msg
    = NoMsg
    | WindowResizes Device.Size
    | DragDropMsg DragDropMsg
    | LoadExampleImages (List String)
    | ImageDecoded { id : String, img : Value }
    | KeyDown RawKey
    | ClickPreviousImage
    | ClickNextImage
    | ZoomMsg ZoomMsg
    | ViewImgMsg ViewImgMsg
    | ParamsMsg ParamsMsg
    | ParamsInfoMsg ParamsInfoMsg
    | NavigationMsg NavigationMsg
    | GetScrollPosThenNavigationMsg NavigationMsg
    | ReturnHome
    | PointerMsg PointerMsg
    | RunAlgorithm Parameters
    | StopRunning
    | UpdateRunStep { step : String, progress : Maybe Int }
    | Log { lvl : Int, content : String }
    | ClearLogs
    | GotScrollPos (Result Browser.Dom.Error Browser.Dom.Viewport)
    | VerbosityChange Float
    | ScrollLogsToEnd
    | ToggleAutoScroll Bool
    | ReceiveCroppedImages (List { id : String, img : Value })
    | SaveRegisteredImages


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
    | ChangeMaxVerbosity String
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
    | ToggleInfoMaxVerbosity Bool
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


type LogsState
    = ErrorLogs
    | WarningLogs
    | NoLogs
    | RegularLogs


logsStatus : List { lvl : Int, content : String } -> LogsState
logsStatus logs =
    case List.minimum (List.map .lvl logs) of
        Nothing ->
            NoLogs

        Just 0 ->
            ErrorLogs

        Just 1 ->
            WarningLogs

        Just _ ->
            RegularLogs


subscriptions : Model -> Sub Msg
subscriptions model =
    case model.state of
        Home _ ->
            Sub.batch [ resizes WindowResizes, log Log, imageDecoded ImageDecoded ]

        Loading _ ->
            Sub.batch [ resizes WindowResizes, log Log, imageDecoded ImageDecoded ]

        LoadingError ->
            Sub.batch [ resizes WindowResizes, log Log, imageDecoded ImageDecoded ]

        ViewImgs _ ->
            Sub.batch [ resizes WindowResizes, log Log, receiveCroppedImages ReceiveCroppedImages, updateRunStep UpdateRunStep, Keyboard.downs KeyDown ]

        Config _ ->
            Sub.batch [ resizes WindowResizes, log Log, receiveCroppedImages ReceiveCroppedImages, updateRunStep UpdateRunStep ]

        Registration _ ->
            Sub.batch [ resizes WindowResizes, log Log, receiveCroppedImages ReceiveCroppedImages, updateRunStep UpdateRunStep, Keyboard.downs KeyDown ]

        Logs _ ->
            Sub.batch [ resizes WindowResizes, log Log, receiveCroppedImages ReceiveCroppedImages, updateRunStep UpdateRunStep ]


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case ( msg, model.state ) of
        ( NoMsg, _ ) ->
            ( model, Cmd.none )

        ( WindowResizes size, _ ) ->
            ( { model
                | device = Device.classify size
                , viewer = Viewer.resize ( size.width, size.height - toFloat (headerHeight + progressBarHeight) ) model.viewer
                , registeredViewer = Viewer.resize ( size.width, size.height - toFloat (headerHeight + progressBarHeight) ) model.registeredViewer
              }
            , Cmd.none
            )

        ( DragDropMsg (DragOver _ _), Home _ ) ->
            ( { model | state = Home DraggingSomeFiles }, Cmd.none )

        ( DragDropMsg (Drop file otherFiles), _ ) ->
            let
                imageFiles =
                    List.filter (\f -> String.startsWith "image" f.mime) (file :: otherFiles)

                names =
                    Set.fromList (List.map .name imageFiles)

                ( newState, cmd, errorLogs ) =
                    if List.isEmpty otherFiles then
                        ( LoadingError
                        , Cmd.none
                        , [ { lvl = 0, content = "Only 1 image was selected. Please pick at least 2." } ]
                        )

                    else
                        ( Loading { names = names, loaded = Dict.empty }
                        , decodeImages (List.map File.encode imageFiles)
                        , []
                        )
            in
            ( { model | state = newState, notSeenLogs = errorLogs }
            , cmd
            )

        ( DragDropMsg DragLeave, Home _ ) ->
            ( { model | state = Home Idle }, Cmd.none )

        ( LoadExampleImages urls, _ ) ->
            ( { model | state = Loading { names = Set.fromList urls, loaded = Dict.empty } }
            , loadImagesFromUrls urls
            )

        ( ImageDecoded ({ id } as imgValue), Loading { names, loaded } ) ->
            let
                newLoaded =
                    case imageFromValue imgValue of
                        Nothing ->
                            -- Should never happen
                            loaded

                        Just image ->
                            Dict.insert id image loaded

                updatedLoadingState =
                    { names = names
                    , loaded = newLoaded
                    }

                oldParamsForm =
                    model.paramsForm
            in
            if Set.size names == Dict.size newLoaded then
                case Dict.values newLoaded of
                    [] ->
                        -- This should be impossible, there must be at least 1 image
                        ( { model | state = Home Idle }, Cmd.none )

                    firstImage :: otherImages ->
                        ( { model
                            | state = ViewImgs { images = Pivot.fromCons firstImage otherImages }
                            , viewer = Viewer.fitImage 1.0 ( toFloat firstImage.width, toFloat firstImage.height ) model.viewer
                            , paramsForm = { oldParamsForm | crop = CropForm.withSize firstImage.width firstImage.height }
                            , imagesCount = Set.size names
                          }
                        , Cmd.none
                        )

            else
                ( { model | state = Loading updatedLoadingState }, Cmd.none )

        ( KeyDown rawKey, ViewImgs { images } ) ->
            case Keyboard.navigationKey rawKey of
                Just Keyboard.ArrowRight ->
                    ( { model | state = ViewImgs { images = goToNextImage images } }, Cmd.none )

                Just Keyboard.ArrowLeft ->
                    ( { model | state = ViewImgs { images = goToPreviousImage images } }, Cmd.none )

                _ ->
                    ( model, Cmd.none )

        ( KeyDown rawKey, Registration _ ) ->
            case Keyboard.navigationKey rawKey of
                Just Keyboard.ArrowRight ->
                    ( { model | registeredImages = Maybe.map goToNextImage model.registeredImages }, Cmd.none )

                Just Keyboard.ArrowLeft ->
                    ( { model | registeredImages = Maybe.map goToPreviousImage model.registeredImages }, Cmd.none )

                _ ->
                    ( model, Cmd.none )

        ( ParamsMsg paramsMsg, Config _ ) ->
            ( updateParams paramsMsg model, Cmd.none )

        ( ParamsInfoMsg paramsInfoMsg, Config _ ) ->
            ( { model | paramsInfo = updateParamsInfo paramsInfoMsg model.paramsInfo }, Cmd.none )

        ( NavigationMsg GoToPageLogs, ViewImgs data ) ->
            ( goTo GoToPageLogs data model, scrollLogsToPos model.scrollPos )

        ( NavigationMsg navMsg, ViewImgs data ) ->
            ( goTo navMsg data model, Cmd.none )

        ( NavigationMsg GoToPageLogs, Config data ) ->
            ( goTo GoToPageLogs data model, scrollLogsToPos model.scrollPos )

        ( NavigationMsg navMsg, Config data ) ->
            ( goTo navMsg data model, Cmd.none )

        ( NavigationMsg GoToPageLogs, Registration data ) ->
            ( goTo GoToPageLogs data model, scrollLogsToPos model.scrollPos )

        ( NavigationMsg navMsg, Registration data ) ->
            ( goTo navMsg data model, Cmd.none )

        ( NavigationMsg navMsg, Logs data ) ->
            ( goTo navMsg data model, Cmd.none )

        ( GetScrollPosThenNavigationMsg navMsg, Logs _ ) ->
            ( model
            , Cmd.batch
                [ Task.attempt GotScrollPos (Browser.Dom.getViewportOf "logs")
                , Process.sleep 0
                    |> Task.perform (always (NavigationMsg navMsg))
                ]
            )

        ( ZoomMsg zoomMsg, ViewImgs _ ) ->
            ( { model | viewer = zoomViewer zoomMsg model.viewer }, Cmd.none )

        ( ZoomMsg zoomMsg, Registration _ ) ->
            ( { model | registeredViewer = zoomViewer zoomMsg model.registeredViewer }, Cmd.none )

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
                                    Pivot.getC (Pivot.goToStart images)

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
                                    newCropForm =
                                        snapBBox (BBox left top right bottom) oldParamsForm.crop

                                    newCrop =
                                        CropForm.decoded newCropForm
                                in
                                ( { model
                                    | pointerMode = WaitingDraw
                                    , bboxDrawn = Maybe.map toBBox newCrop
                                    , params = { oldParams | crop = newCrop }
                                    , paramsForm = { oldParamsForm | crop = newCropForm }
                                  }
                                , Cmd.none
                                )

                            else
                                ( { model
                                    | pointerMode = WaitingDraw
                                    , bboxDrawn = Nothing
                                    , params = { oldParams | crop = Nothing }
                                    , paramsForm = { oldParamsForm | crop = CropForm.toggle False oldParamsForm.crop }
                                  }
                                , Cmd.none
                                )

                        Nothing ->
                            ( model, Cmd.none )

                _ ->
                    ( model, Cmd.none )

        ( ViewImgMsg CropCurrentFrame, ViewImgs { images } ) ->
            let
                img =
                    Pivot.getC (Pivot.goToStart images)

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
                    newCropForm =
                        snapBBox (BBox left top right bottom) oldParamsForm.crop

                    newCrop =
                        CropForm.decoded newCropForm
                in
                ( { model
                    | bboxDrawn = Maybe.map toBBox newCrop
                    , params = { oldParams | crop = newCrop }
                    , paramsForm = { oldParamsForm | crop = newCropForm }
                  }
                , Cmd.none
                )

            else
                ( { model
                    | bboxDrawn = Nothing
                    , params = { oldParams | crop = Nothing }
                    , paramsForm = { oldParamsForm | crop = CropForm.toggle False oldParamsForm.crop }
                  }
                , Cmd.none
                )

        ( ViewImgMsg SelectMovingMode, ViewImgs _ ) ->
            ( { model | pointerMode = WaitingMove }, Cmd.none )

        ( ViewImgMsg SelectDrawingMode, ViewImgs _ ) ->
            ( { model | pointerMode = WaitingDraw }, Cmd.none )

        ( ClickPreviousImage, ViewImgs { images } ) ->
            ( { model | state = ViewImgs { images = goToPreviousImage images } }, Cmd.none )

        ( ClickPreviousImage, Registration _ ) ->
            ( { model | registeredImages = Maybe.map goToPreviousImage model.registeredImages }, Cmd.none )

        ( ClickNextImage, ViewImgs { images } ) ->
            ( { model | state = ViewImgs { images = goToNextImage images } }, Cmd.none )

        ( ClickNextImage, Registration _ ) ->
            ( { model | registeredImages = Maybe.map goToNextImage model.registeredImages }, Cmd.none )

        ( RunAlgorithm params, ViewImgs imgs ) ->
            ( runAndSwitchToLogsPage imgs model
            , run (encodeParams params)
            )

        ( RunAlgorithm params, Config imgs ) ->
            ( runAndSwitchToLogsPage imgs model
            , run (encodeParams params)
            )

        ( RunAlgorithm params, Registration imgs ) ->
            ( runAndSwitchToLogsPage imgs model
            , run (encodeParams params)
            )

        ( RunAlgorithm params, Logs imgs ) ->
            ( runAndSwitchToLogsPage imgs model
            , run (encodeParams params)
            )

        ( StopRunning, _ ) ->
            ( { model | runStep = StepNotStarted }, stop () )

        ( UpdateRunStep { step, progress }, _ ) ->
            let
                runStep =
                    case ( model.runStep, step, progress ) of
                        ( _, "Precompute multiresolution pyramid", _ ) ->
                            StepMultiresPyramid

                        ( _, "level", Just lvl ) ->
                            StepLevel lvl

                        ( StepLevel lvl, "iteration", Just iter ) ->
                            StepIteration lvl iter

                        ( StepIteration lvl _, "iteration", Just iter ) ->
                            StepIteration lvl iter

                        ( _, "Reproject", Just im ) ->
                            StepApplying im

                        ( _, "encoding", Just im ) ->
                            StepEncoding im

                        ( _, "done", _ ) ->
                            StepDone

                        ( _, "saving", Just im ) ->
                            StepSaving im

                        _ ->
                            StepNotStarted
            in
            ( { model | runStep = runStep }, Cmd.none )

        ( Log logData, Logs _ ) ->
            let
                newLogs =
                    logData :: model.seenLogs
            in
            if model.autoscroll then
                ( { model | seenLogs = newLogs }, scrollLogsToEndCmd )

            else
                ( { model | seenLogs = newLogs }, Cmd.none )

        ( Log logData, Loading _ ) ->
            let
                newState =
                    if logData.lvl == 0 then
                        LoadingError

                    else
                        model.state
            in
            ( { model | notSeenLogs = logData :: model.notSeenLogs, state = newState }, Cmd.none )

        ( Log logData, _ ) ->
            ( { model | notSeenLogs = logData :: model.notSeenLogs }, Cmd.none )

        ( GotScrollPos (Ok { viewport }), Logs _ ) ->
            ( { model | scrollPos = viewport.y }, Cmd.none )

        ( VerbosityChange floatVerbosity, _ ) ->
            ( { model | verbosity = round floatVerbosity }, Cmd.none )

        ( ScrollLogsToEnd, Logs _ ) ->
            ( model, scrollLogsToEndCmd )

        ( ToggleAutoScroll activate, _ ) ->
            if activate then
                ( { model | autoscroll = True }, scrollLogsToEndCmd )

            else
                ( { model | autoscroll = False }, Cmd.none )

        ( ReceiveCroppedImages croppedImages, _ ) ->
            case List.filterMap imageFromValue croppedImages of
                [] ->
                    ( model, Cmd.none )

                firstImage :: otherImages ->
                    ( { model
                        | registeredImages = Just (Pivot.fromCons firstImage otherImages)
                        , registeredViewer = Viewer.fitImage 1.0 ( toFloat firstImage.width, toFloat firstImage.height ) model.registeredViewer
                      }
                    , Cmd.none
                    )

        ( PointerMsg pointerMsg, Registration _ ) ->
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
                        | registeredViewer = Viewer.pan ( newX - x, newY - y ) model.registeredViewer
                        , pointerMode = PointerMovingFromClientCoords ( newX, newY )
                      }
                    , Cmd.none
                    )

                ( PointerUp _, PointerMovingFromClientCoords _ ) ->
                    ( { model | pointerMode = WaitingMove }, Cmd.none )

                _ ->
                    ( model, Cmd.none )

        ( SaveRegisteredImages, _ ) ->
            ( model, saveRegisteredImages model.imagesCount )

        ( ReturnHome, _ ) ->
            ( model, Browser.Navigation.reload )

        ( ClearLogs, _ ) ->
            ( { model
                | seenLogs =
                    [ { content = "Logs cleared"
                      , lvl = 3
                      }
                    ]
                , notSeenLogs = []
              }
            , Cmd.none
            )

        _ ->
            ( model, Cmd.none )


runAndSwitchToLogsPage : { images : Pivot Image } -> Model -> Model
runAndSwitchToLogsPage imgs model =
    goTo GoToPageLogs imgs { model | registeredImages = Nothing, runStep = StepNotStarted }


scrollLogsToEndCmd : Cmd Msg
scrollLogsToEndCmd =
    Task.attempt (\_ -> NoMsg) (Browser.Dom.setViewportOf "logs" 0 1.0e14)


scrollLogsToPos : Float -> Cmd Msg
scrollLogsToPos pos =
    Task.attempt (\_ -> NoMsg) (Browser.Dom.setViewportOf "logs" 0 pos)


imageFromValue : { id : String, img : Value } -> Maybe Image
imageFromValue { id, img } =
    case Canvas.Texture.fromDomImage img of
        Nothing ->
            Nothing

        Just texture ->
            let
                imgSize =
                    Canvas.Texture.dimensions texture
            in
            Just
                { id = id
                , texture = texture
                , width = round imgSize.width
                , height = round imgSize.height
                }


goToPreviousImage : Pivot Image -> Pivot Image
goToPreviousImage images =
    Maybe.withDefault (Pivot.goToEnd images) (Pivot.goL images)


goToNextImage : Pivot Image -> Pivot Image
goToNextImage images =
    Maybe.withDefault (Pivot.goToStart images) (Pivot.goR images)


toBBox : Crop -> BBox
toBBox { left, top, right, bottom } =
    { left = toFloat left
    , top = toFloat top
    , right = toFloat right
    , bottom = toFloat bottom
    }


{-| Restrict coordinates of a drawn bounding box to the image dimension.
-}
snapBBox : BBox -> CropForm.State -> CropForm.State
snapBBox { left, top, right, bottom } state =
    let
        maxRight =
            -- Should never be Nothing
            Maybe.withDefault 0 state.right.max

        maxBottom =
            -- Should never be Nothing
            Maybe.withDefault 0 state.bottom.max

        leftCrop =
            round (max 0 left)

        topCrop =
            round (max 0 top)

        rightCrop =
            min (round right) maxRight

        bottomCrop =
            min (round bottom) maxBottom
    in
    CropForm.toggle True state
        |> CropForm.updateLeft (String.fromInt leftCrop)
        |> CropForm.updateTop (String.fromInt topCrop)
        |> CropForm.updateRight (String.fromInt rightCrop)
        |> CropForm.updateBottom (String.fromInt bottomCrop)


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


goTo : NavigationMsg -> { images : Pivot Image } -> Model -> Model
goTo msg data model =
    case msg of
        GoToPageImages ->
            { model | state = ViewImgs data, pointerMode = WaitingMove }

        GoToPageConfig ->
            { model | state = Config data }

        GoToPageRegistration ->
            { model | state = Registration data, pointerMode = WaitingMove }

        GoToPageLogs ->
            { model
                | state = Logs data
                , seenLogs = List.concat [ model.notSeenLogs, model.seenLogs ]
                , notSeenLogs = []
            }


updateParams : ParamsMsg -> Model -> Model
updateParams msg ({ params, paramsForm } as model) =
    case msg of
        ToggleEqualize equalize ->
            { model | params = { params | equalize = equalize } }

        ChangeMaxVerbosity str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.maxVerbosity

                updatedForm =
                    { paramsForm | maxVerbosity = updatedField }
            in
            case updatedField.decodedInput of
                Ok maxVerbosity ->
                    { model
                        | params = { params | maxVerbosity = maxVerbosity }
                        , paramsForm = updatedForm
                    }

                Err _ ->
                    { model | paramsForm = updatedForm }

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
                ( True, Just crop ) ->
                    { model
                        | params = { params | crop = Just crop }
                        , paramsForm = { paramsForm | crop = newCropForm }
                        , bboxDrawn = Just (toBBox crop)
                    }

                _ ->
                    { model
                        | params = { params | crop = Nothing }
                        , paramsForm = { paramsForm | crop = newCropForm }
                        , bboxDrawn = Nothing
                    }

        ChangeCropLeft str ->
            changeCropSide (CropForm.updateLeft str) model

        ChangeCropTop str ->
            changeCropSide (CropForm.updateTop str) model

        ChangeCropRight str ->
            changeCropSide (CropForm.updateRight str) model

        ChangeCropBottom str ->
            changeCropSide (CropForm.updateBottom str) model


changeCropSide : (CropForm.State -> CropForm.State) -> Model -> Model
changeCropSide updateSide model =
    let
        params =
            model.params

        paramsForm =
            model.paramsForm

        newCropForm =
            updateSide paramsForm.crop

        newCrop =
            CropForm.decoded newCropForm
    in
    { model
        | params = { params | crop = newCrop }
        , paramsForm = { paramsForm | crop = newCropForm }
        , bboxDrawn = Maybe.map toBBox newCrop
    }


updateParamsInfo : ParamsInfoMsg -> ParametersToggleInfo -> ParametersToggleInfo
updateParamsInfo msg toggleInfo =
    case msg of
        ToggleInfoCrop visible ->
            { toggleInfo | crop = visible }

        ToggleInfoMaxIterations visible ->
            { toggleInfo | maxIterations = visible }

        ToggleInfoMaxVerbosity visible ->
            { toggleInfo | maxVerbosity = visible }

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

        LoadingError ->
            viewLoadingError model

        ViewImgs { images } ->
            viewImgs model images

        Config _ ->
            viewConfig model

        Registration _ ->
            viewRegistration model

        Logs _ ->
            viewLogs model



-- Header


{-| WARNING: this has to be kept consistent with the text size in the header
-}
headerHeight : Int
headerHeight =
    40


headerBar : List (Element Msg) -> Element Msg
headerBar pages =
    Element.row
        [ height (Element.px headerHeight)
        , centerX
        ]
        pages


headerTab : String -> Maybe Msg -> Element Msg
headerTab label msg =
    headerTabWithAttributes label msg []


headerTabWithAttributes : String -> Maybe Msg -> List (Element.Attribute Msg) -> Element Msg
headerTabWithAttributes label msg otherAttributes =
    let
        bgColor =
            if msg == Nothing then
                Style.almostWhite

            else
                Style.white
    in
    Element.Input.button (otherAttributes ++ baseTabAttributes bgColor)
        { onPress = msg
        , label = Element.text label
        }


baseTabAttributes : Element.Color -> List (Element.Attribute msg)
baseTabAttributes bgColor =
    [ Element.Background.color bgColor
    , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
    , padding 10
    , height (Element.px headerHeight)
    ]


registrationHeaderTab : Maybe Msg -> Maybe (Pivot Image) -> Element Msg
registrationHeaderTab msg registeredImages =
    let
        otherAttributes =
            if registeredImages == Nothing then
                []

            else
                [ Element.inFront (Element.el [ alignRight, padding 2 ] (littleDot "green" |> Element.html)) ]
    in
    headerTabWithAttributes "Registration" msg otherAttributes


logsHeaderTab : Maybe Msg -> List { lvl : Int, content : String } -> Element Msg
logsHeaderTab msg logs =
    let
        logsState =
            logsStatus logs

        fillColor =
            case logsState of
                -- Style.errorColor
                ErrorLogs ->
                    "rgb(180,50,50)"

                -- Style.warningColor
                WarningLogs ->
                    "rgb(220,120,50)"

                -- Style.darkGrey
                _ ->
                    "rgb(50,50,50)"

        otherAttributes =
            case logsState of
                NoLogs ->
                    []

                _ ->
                    [ Element.inFront (Element.el [ alignRight, padding 2 ] (littleDot fillColor |> Element.html)) ]
    in
    headerTabWithAttributes "Logs" msg otherAttributes


littleDot : String -> Html msg
littleDot fillColor =
    Svg.svg
        [ Svg.Attributes.viewBox "0 0 10 10"
        , Svg.Attributes.width "10"
        , Svg.Attributes.height "10"
        ]
        [ Svg.circle
            [ Svg.Attributes.cx "5"
            , Svg.Attributes.cy "5"
            , Svg.Attributes.r "5"
            , Svg.Attributes.fill fillColor
            ]
            []
        ]



-- Run progress


{-| WARNING: this has to be kept consistent with the viewer size
-}
progressBarHeight : Int
progressBarHeight =
    38


runProgressBar : Model -> Element Msg
runProgressBar model =
    let
        progressBarRunButton =
            case model.runStep of
                StepNotStarted ->
                    runButton "Run ▶" model.params model.paramsForm

                StepDone ->
                    runButton "Rerun ▶" model.params model.paramsForm

                _ ->
                    Element.none

        progressBarStopButton =
            if model.runStep == StepNotStarted || model.runStep == StepDone then
                Element.none

            else
                stopButton

        progressBarSaveButton =
            if model.runStep == StepDone then
                saveButton

            else
                Element.none
    in
    Element.el
        [ width fill
        , height (Element.px progressBarHeight)
        , Element.Font.size 12
        , Element.behindContent (progressBar Style.almostWhite 1.0)
        , Element.behindContent (progressBar Style.runProgressColor <| estimateProgress model)
        , Element.inFront progressBarRunButton
        , Element.inFront progressBarStopButton
        , Element.inFront progressBarSaveButton
        ]
        (Element.el [ centerX, centerY ] (Element.text <| progressMessage model))


runButton : String -> Parameters -> ParametersForm -> Element Msg
runButton content params paramsForm =
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
            { onPress = Just (RunAlgorithm params), label = Element.text content }

    else
        Element.Input.button
            [ centerX
            , padding 12
            , Element.Border.solid
            , Element.Border.width 1
            , Element.Border.rounded 4
            , Element.Font.color Style.lightGrey
            ]
            { onPress = Nothing, label = Element.text content }


isOk : Result err ok -> Bool
isOk result =
    case result of
        Err _ ->
            False

        Ok _ ->
            True


stopButton : Element Msg
stopButton =
    Element.Input.button
        [ alignRight
        , padding 12
        , Element.Border.solid
        , Element.Border.width 1
        , Element.Border.rounded 4
        ]
        { onPress = Just StopRunning
        , label = Element.text "Stop!"
        }


saveButton : Element Msg
saveButton =
    Element.Input.button
        [ alignRight
        , padding 12
        , Element.Border.solid
        , Element.Border.width 1
        , Element.Border.rounded 4
        ]
        { onPress = Just SaveRegisteredImages
        , label = Element.text "Save registered images"
        }


progressMessage : Model -> String
progressMessage model =
    case model.runStep of
        StepNotStarted ->
            ""

        StepMultiresPyramid ->
            "Building multi-resolution pyramid"

        StepLevel level ->
            "Registering at level " ++ String.fromInt level

        StepIteration level iter ->
            "Registering at level " ++ String.fromInt level ++ "    iteration " ++ String.fromInt iter ++ " / " ++ String.fromInt model.params.maxIterations

        StepApplying img ->
            "Applying warp to cropped image " ++ String.fromInt img ++ " / " ++ String.fromInt model.imagesCount

        StepEncoding img ->
            "Encoding registered cropped image " ++ String.fromInt img ++ " / " ++ String.fromInt model.imagesCount

        StepDone ->
            ""

        StepSaving img ->
            "Warping and encoding image " ++ String.fromInt img ++ " / " ++ String.fromInt model.imagesCount


estimateProgress : Model -> Float
estimateProgress model =
    let
        subprogress n nCount =
            toFloat n / toFloat nCount

        lvlCount =
            model.params.levels

        levelProgress lvl =
            subprogress (lvlCount - lvl - 1) lvlCount
    in
    case model.runStep of
        StepNotStarted ->
            0.0

        -- 0 to 10% for pyramid
        StepMultiresPyramid ->
            0.0

        -- Say 10% to 80% to share for all levels
        -- We can imagine each next level 2 times slower (approximation)
        StepLevel level ->
            0.1 + 0.7 * levelProgress level

        StepIteration level iter ->
            0.1 + 0.7 * levelProgress level + 0.7 / toFloat lvlCount * subprogress iter model.params.maxIterations

        -- Say 80% to 90% for applying registration to cropped images
        StepApplying img ->
            0.8 + 0.1 * subprogress img model.imagesCount

        -- Say 90% to 100% for encoding the registered cropped images
        StepEncoding img ->
            0.9 + 0.1 * subprogress img model.imagesCount

        StepDone ->
            1.0

        StepSaving img ->
            subprogress img model.imagesCount


progressBar : Element.Color -> Float -> Element Msg
progressBar color progressRatio =
    let
        scaleX =
            "scaleX(" ++ String.fromFloat progressRatio ++ ")"
    in
    Element.el
        [ width fill
        , height fill
        , Element.Background.color color
        , Element.htmlAttribute (Html.Attributes.style "transform-origin" "top left")
        , Element.htmlAttribute (Html.Attributes.style "transform" scaleX)
        ]
        Element.none



-- Logs


viewLogs : Model -> Element Msg
viewLogs ({ autoscroll, verbosity, seenLogs, notSeenLogs, registeredImages } as model) =
    Element.column [ width fill, height fill ]
        [ headerBar
            [ headerTab "Images" (Just (GetScrollPosThenNavigationMsg GoToPageImages))
            , headerTab "Config" (Just (GetScrollPosThenNavigationMsg GoToPageConfig))
            , registrationHeaderTab (Just (GetScrollPosThenNavigationMsg GoToPageRegistration)) registeredImages
            , logsHeaderTab Nothing notSeenLogs
            ]
        , runProgressBar model
        , Element.column [ width fill, height fill, paddingXY 0 18, spacing 18 ]
            [ Element.el [ centerX ] (verbositySlider verbosity)
            , Element.row [ centerX, spacing 18 ]
                [ Element.el [ centerY ] (Element.text "auto scroll:")
                , Element.el [ centerY ] (Element.text "off")
                , toggle ToggleAutoScroll autoscroll 24 "autoscroll"
                , Element.el [ centerY ] (Element.text "on")
                ]
            , clearLogsButton
            , Element.column
                [ padding 18
                , height fill
                , width fill
                , centerX
                , Style.fontMonospace
                , Element.Font.size 18
                , Element.scrollbars
                , Element.htmlAttribute (Html.Attributes.id "logs")
                ]
                (List.filter (\l -> l.lvl <= verbosity) seenLogs
                    |> List.reverse
                    |> List.map viewLog
                )
            ]
        ]


clearLogsButton : Element Msg
clearLogsButton =
    Element.row [ centerX, spacing 15 ]
        [ Element.Input.button
            [ Element.Background.color Style.almostWhite
            , padding 10
            ]
            { onPress = Just ClearLogs
            , label = Icon.trash 24
            }
        , Element.el [ centerY ] (Element.text "Clear logs")
        ]


viewLog : { lvl : Int, content : String } -> Element msg
viewLog { lvl, content } =
    case lvl of
        0 ->
            Element.el
                [ Element.Font.color Style.errorColor
                , paddingXY 0 12
                , Element.onLeft
                    (Element.el [ height fill, paddingXY 0 4 ]
                        (Element.el
                            [ height fill
                            , width (Element.px 4)
                            , Element.Background.color Style.errorColor
                            , Element.moveLeft 6
                            ]
                            Element.none
                        )
                    )
                ]
                (Element.text content)

        1 ->
            Element.el
                [ Element.Font.color Style.warningColor
                , paddingXY 0 12
                , Element.onLeft
                    (Element.el [ height fill, paddingXY 0 4 ]
                        (Element.el
                            [ height fill
                            , width (Element.px 4)
                            , Element.Background.color Style.warningColor
                            , Element.moveLeft 6
                            ]
                            Element.none
                        )
                    )
                ]
                (Element.text content)

        _ ->
            Element.text content


verbositySlider : Int -> Element Msg
verbositySlider verbosity =
    let
        thumbSize =
            32

        circle color size =
            [ Element.Border.color color
            , width (Element.px size)
            , height (Element.px size)
            , Element.Border.rounded (size // 2)
            , Element.Border.width 2
            ]
    in
    Element.Input.slider
        [ width (Element.px 200)
        , height (Element.px thumbSize)
        , spacing 18

        -- Here is where we're creating/styling the "track"
        , Element.behindContent <|
            Element.row [ width fill, centerY ]
                [ Element.el (circle Style.errorColor thumbSize) Element.none
                , Element.el [ width fill ] Element.none
                , Element.el (circle Style.warningColor thumbSize) Element.none
                , Element.el [ width fill ] Element.none
                , Element.el (circle Style.lightGrey thumbSize) Element.none
                , Element.el [ width fill ] Element.none
                , Element.el (circle Style.lightGrey thumbSize) Element.none
                , Element.el [ width fill ] Element.none
                , Element.el (circle Style.lightGrey thumbSize) Element.none
                ]
        ]
        { onChange = VerbosityChange
        , label =
            Element.Input.labelLeft [ centerY, Element.Font.size 18 ]
                (Element.el [] (Element.text "  Verbosity\n(min -> max)"))
        , min = 0
        , max = 4
        , step = Just 1
        , value = toFloat verbosity

        -- Here is where we're creating the "thumb"
        , thumb =
            let
                color =
                    if verbosity == 0 then
                        Style.errorColor

                    else if verbosity == 1 then
                        Style.warningColor

                    else
                        Style.lightGrey
            in
            Element.Input.thumb
                [ Element.Background.color color
                , width (Element.px thumbSize)
                , height (Element.px thumbSize)
                , Element.Border.rounded (thumbSize // 2)
                ]
        }



-- Registration


viewRegistration : Model -> Element Msg
viewRegistration ({ registeredImages, registeredViewer, notSeenLogs } as model) =
    Element.column [ width fill, height fill ]
        [ headerBar
            [ headerTab "Images" (Just (NavigationMsg GoToPageImages))
            , headerTab "Config" (Just (NavigationMsg GoToPageConfig))
            , registrationHeaderTab Nothing registeredImages
            , logsHeaderTab (Just (NavigationMsg GoToPageLogs)) notSeenLogs
            ]
        , runProgressBar model
        , Element.html <|
            Html.node "style"
                []
                [ Html.text ".pixelated { image-rendering: pixelated; image-rendering: crisp-edges; }" ]
        , case registeredImages of
            Nothing ->
                Element.el [ centerX, centerY ]
                    (Element.text "Registration not done yet")

            Just images ->
                let
                    img =
                        Pivot.getC images

                    clickButton alignment msg title icon =
                        Element.Input.button
                            [ padding 6
                            , alignment
                            , Element.Background.color (Element.rgba255 255 255 255 0.8)
                            , Element.Font.color Style.black
                            , Element.htmlAttribute <| Html.Attributes.style "box-shadow" "none"
                            , Element.htmlAttribute <| Html.Attributes.title title
                            ]
                            { onPress = Just msg
                            , label = icon 32
                            }

                    buttonsRow =
                        Element.row [ centerX ]
                            [ clickButton centerX (ZoomMsg (ZoomFit img)) "Fit zoom to image" Icon.zoomFit
                            , clickButton centerX (ZoomMsg ZoomOut) "Zoom out" Icon.zoomOut
                            , clickButton centerX (ZoomMsg ZoomIn) "Zoom in" Icon.zoomIn
                            ]

                    ( viewerWidth, viewerHeight ) =
                        registeredViewer.size

                    clearCanvas : Canvas.Renderable
                    clearCanvas =
                        Canvas.clear ( 0, 0 ) viewerWidth viewerHeight

                    renderedImage : Canvas.Renderable
                    renderedImage =
                        Canvas.texture
                            [ Viewer.Canvas.transform registeredViewer
                            , Canvas.Settings.Advanced.imageSmoothing False
                            ]
                            ( 0, 0 )
                            img.texture

                    canvasViewer =
                        Canvas.toHtml ( round viewerWidth, round viewerHeight )
                            [ Html.Attributes.id "theCanvas"
                            , Html.Attributes.style "display" "block"
                            , Wheel.onWheel (zoomWheelMsg registeredViewer)
                            , msgOn "pointerdown" (Json.Decode.map (PointerMsg << PointerDownRaw) Json.Decode.value)
                            , Pointer.onUp (\e -> PointerMsg (PointerUp e.pointer.offsetPos))
                            , Html.Attributes.style "touch-action" "none"
                            , Html.Events.preventDefaultOn "pointermove" <|
                                Json.Decode.map (\coords -> ( PointerMsg (PointerMove coords), True )) <|
                                    Json.Decode.map2 Tuple.pair
                                        (Json.Decode.field "clientX" Json.Decode.float)
                                        (Json.Decode.field "clientY" Json.Decode.float)
                            ]
                            [ clearCanvas, renderedImage ]
                in
                Element.el
                    [ Element.inFront buttonsRow
                    , Element.inFront
                        (Element.row [ alignBottom, width fill ]
                            [ clickButton alignLeft ClickPreviousImage "Previous image" Icon.arrowLeftCircle
                            , clickButton alignRight ClickNextImage "Next image" Icon.arrowRightCircle
                            ]
                        )
                    , Element.clip
                    , height fill
                    ]
                    (Element.html canvasViewer)
        ]



-- Parameters config


viewConfig : Model -> Element Msg
viewConfig ({ params, paramsForm, paramsInfo, notSeenLogs, registeredImages } as model) =
    Element.column [ width fill, height fill ]
        [ headerBar
            [ headerTab "Images" (Just (NavigationMsg GoToPageImages))
            , headerTab "Config" Nothing
            , registrationHeaderTab (Just (NavigationMsg GoToPageRegistration)) registeredImages
            , logsHeaderTab (Just (NavigationMsg GoToPageLogs)) notSeenLogs
            ]
        , runProgressBar model
        , Element.column [ width fill, height fill, Element.scrollbars ]
            [ Element.column [ paddingXY 20 32, spacing 32, centerX ]
                [ -- Title
                  Element.el [ Element.Font.center, Element.Font.size 32 ] (Element.text "Algorithm parameters")

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

                -- Maximum verbosity
                , Element.column [ spacing 10 ]
                    [ Element.row [ spacing 10 ]
                        [ Element.text "Maximum verbosity:"
                        , Element.Input.checkbox []
                            { onChange = ParamsInfoMsg << ToggleInfoMaxVerbosity
                            , icon = infoIcon
                            , checked = paramsInfo.maxVerbosity
                            , label = Element.Input.labelHidden "Show detail info about the maximum verbosity."
                            }
                        ]
                    , moreInfo paramsInfo.maxVerbosity "Maximum verbosity of logs that can appear in the Logs tab. Setting this higher than its default value enables a very detailed log trace at the price of performance degradations."
                    , Element.text ("(default to " ++ String.fromInt defaultParams.maxVerbosity ++ ")")
                    , intInput paramsForm.maxVerbosity (ParamsMsg << ChangeMaxVerbosity) "Maximum verbosity"
                    , displayIntErrors paramsForm.maxVerbosity.decodedInput
                    ]
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


viewImgs : Model -> Pivot Image -> Element Msg
viewImgs ({ pointerMode, bboxDrawn, viewer, notSeenLogs, registeredImages } as model) images =
    let
        img =
            Pivot.getC images

        clickButton alignment abled msg title icon =
            let
                strokeColor =
                    if abled then
                        Style.black

                    else
                        Style.lightGrey
            in
            Element.Input.button
                [ padding 6
                , alignment
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
                [ clickButton centerX True (ZoomMsg (ZoomFit img)) "Fit zoom to image" Icon.zoomFit
                , clickButton centerX True (ZoomMsg ZoomOut) "Zoom out" Icon.zoomOut
                , clickButton centerX True (ZoomMsg ZoomIn) "Zoom in" Icon.zoomIn
                , modeButton isMovingMode (ViewImgMsg SelectMovingMode) "Move mode" Icon.move
                , Element.el [ width (Element.maximum 100 fill) ] Element.none
                , modeButton (not isMovingMode) (ViewImgMsg SelectDrawingMode) "Draw the cropped working area as a bounding box" Icon.boundingBox
                , clickButton centerX True (ViewImgMsg CropCurrentFrame) "Set the cropped working area to the current frame" Icon.maximize
                ]

        ( viewerWidth, viewerHeight ) =
            viewer.size

        clearCanvas : Canvas.Renderable
        clearCanvas =
            Canvas.clear ( 0, 0 ) viewerWidth viewerHeight

        renderedImage : Canvas.Renderable
        renderedImage =
            Canvas.texture
                [ Viewer.Canvas.transform viewer
                , Canvas.Settings.Advanced.imageSmoothing False
                ]
                ( 0, 0 )
                img.texture

        renderedBbox =
            case bboxDrawn of
                Nothing ->
                    Canvas.shapes [] []

                Just { left, top, right, bottom } ->
                    let
                        bboxWidth =
                            right - left

                        bboxHeight =
                            bottom - top

                        strokeWidth =
                            viewer.scale * 2
                    in
                    Canvas.shapes
                        [ Canvas.Settings.fill (Color.rgba 1 1 1 0.3)
                        , Canvas.Settings.stroke Color.red
                        , Canvas.Settings.Line.lineWidth strokeWidth
                        , Viewer.Canvas.transform viewer
                        ]
                        [ Canvas.rect ( left, top ) bboxWidth bboxHeight ]

        canvasViewer =
            Canvas.toHtml ( round viewerWidth, round viewerHeight )
                [ Html.Attributes.id "theCanvas"
                , Html.Attributes.style "display" "block"
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
                [ clearCanvas, renderedImage, renderedBbox ]
    in
    Element.column [ height fill ]
        [ headerBar
            [ headerTab "Images" Nothing
            , headerTab "Config" (Just (NavigationMsg GoToPageConfig))
            , registrationHeaderTab (Just (NavigationMsg GoToPageRegistration)) registeredImages
            , logsHeaderTab (Just (NavigationMsg GoToPageLogs)) notSeenLogs
            ]
        , runProgressBar model
        , Element.html <|
            Html.node "style"
                []
                [ Html.text ".pixelated { image-rendering: pixelated; image-rendering: crisp-edges; }" ]
        , Element.el
            [ Element.inFront buttonsRow
            , Element.inFront
                (Element.row [ alignBottom, width fill ]
                    [ clickButton alignLeft True ClickPreviousImage "Previous image" Icon.arrowLeftCircle
                    , clickButton alignRight True ClickNextImage "Next image" Icon.arrowRightCircle
                    ]
                )
            , Element.clip
            , height fill
            ]
            (Element.html canvasViewer)
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
        , Element.Input.button
            [ Element.Background.color Style.almostWhite
            , Element.Border.dotted
            , Element.Border.width 2
            , padding 16
            , centerX
            ]
            { onPress = Just ReturnHome
            , label = Element.text "Woops, stop and reload the page"
            }
        ]


viewLoadingError : Model -> Element Msg
viewLoadingError model =
    Element.column [ padding 20, width fill, height fill, spacing 48 ]
        [ viewTitle
        , Element.paragraph [ width (Element.maximum 400 fill), centerX ]
            [ Element.text "An unrecoverable error occured while loading the images, please reload the page." ]
        , Element.Input.button
            [ Element.Background.color Style.almostWhite
            , Element.Border.dotted
            , Element.Border.width 2
            , padding 16
            , centerX
            ]
            { onPress = Just ReturnHome
            , label = Element.text "Woops, reload the page"
            }
        , Element.column
            [ padding 18
            , height fill
            , width fill
            , centerX
            , Style.fontMonospace
            , Element.Font.size 18
            , Element.scrollbars
            , Element.htmlAttribute (Html.Attributes.id "logs")
            ]
            (List.filter (\l -> l.lvl <= 0) model.notSeenLogs
                |> List.reverse
                |> List.map viewLog
            )
        ]


loadBar : Int -> Int -> Element msg
loadBar loaded total =
    let
        barLength =
            (325 - 2 * 4) * loaded // total
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
            Element.row [ centerX ]
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

        useDirectlyProvided =
            Element.paragraph [ centerX, Element.Font.center ]
                [ Element.text "You can also directly use "
                , Element.Input.button [ Element.Font.underline ]
                    { onPress =
                        Just
                            (LoadExampleImages
                                [ "/img/bd_caen/01.jpg"
                                , "/img/bd_caen/02.jpg"
                                , "/img/bd_caen/03.jpg"
                                , "/img/bd_caen/04.jpg"
                                , "/img/bd_caen/05.jpg"
                                , "/img/bd_caen/06.jpg"
                                ]
                            )
                    , label = Element.text "this example set of 6 images"
                    }
                ]
    in
    Element.el [ width fill, height fill ]
        (Element.column [ centerX, centerY, spacing 32 ]
            [ Element.el (dropIconBorderAttributes borderStyle) (Icon.arrowDown 48)
            , dropOrLoadText
            , useDirectlyProvided
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
