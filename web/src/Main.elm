port module Main exposing (main)

import Browser
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
import Icon
import Json.Decode exposing (Value)
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
    { crop : CropForm
    , maxIterations : NumberInput.Field Int NumberInput.IntError
    , convergenceThreshold : NumberInput.Field Float NumberInput.FloatError
    , levels : NumberInput.Field Int NumberInput.IntError
    , sparse : NumberInput.Field Float NumberInput.FloatError
    , lambda : NumberInput.Field Float NumberInput.FloatError
    , rho : NumberInput.Field Float NumberInput.FloatError
    }


type alias CropForm =
    { active : Bool
    , left : NumberInput.Field Int NumberInput.IntError
    , top : NumberInput.Field Int NumberInput.IntError
    , right : NumberInput.Field Int NumberInput.IntError
    , bottom : NumberInput.Field Int NumberInput.IntError
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
      }
    , Cmd.none
    )


initialState : State
initialState =
    -- Home Idle
    -- Config { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }
    ViewImgs { images = Pivot.fromCons (Image "ferris" "https://opensource.com/sites/default/files/styles/teaser-wide/public/lead-images/rust_programming_crab_sea.png?itok=Nq53PhmO" 249 140) [] }


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
    { crop = defaultCropForm 1920 1080
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


defaultCropForm : Int -> Int -> CropForm
defaultCropForm width height =
    let
        anyInt =
            NumberInput.intDefault
    in
    { active = defaultParams.crop /= Nothing
    , left =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue 0
    , top =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue 0
    , right =
        { anyInt | min = Just 0, max = Just width }
            |> NumberInput.setDefaultIntValue width
    , bottom =
        { anyInt | min = Just 0, max = Just height }
            |> NumberInput.setDefaultIntValue height
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
    | ViewImgMsg ViewImgMsg
    | ParamsMsg ParamsMsg
    | ParamsInfoMsg ParamsInfoMsg
    | NavigationMsg NavigationMsg


type DragDropMsg
    = DragOver File (List File)
    | Drop File (List File)
    | DragLeave


type ViewImgMsg
    = TODO


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
            let
                ( newParams, newParamsForm ) =
                    updateParams paramsMsg ( model.params, model.paramsForm )
            in
            ( { model | params = newParams, paramsForm = newParamsForm }, Cmd.none )

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

        _ ->
            ( model, Cmd.none )


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


updateParams : ParamsMsg -> ( Parameters, ParametersForm ) -> ( Parameters, ParametersForm )
updateParams msg ( params, paramsForm ) =
    case msg of
        ToggleEqualize equalize ->
            ( { params | equalize = equalize }, paramsForm )

        ChangeMaxIter str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.maxIterations

                updatedForm =
                    { paramsForm | maxIterations = updatedField }
            in
            case updatedField.decodedInput of
                Ok maxIterations ->
                    ( { params | maxIterations = maxIterations }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeConvergenceThreshold str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.convergenceThreshold

                updatedForm =
                    { paramsForm | convergenceThreshold = updatedField }
            in
            case updatedField.decodedInput of
                Ok convergenceThreshold ->
                    ( { params | convergenceThreshold = convergenceThreshold }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeLevels str ->
            let
                updatedField =
                    NumberInput.updateInt str paramsForm.levels

                updatedForm =
                    { paramsForm | levels = updatedField }
            in
            case updatedField.decodedInput of
                Ok levels ->
                    ( { params | levels = levels }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeSparse str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.sparse

                updatedForm =
                    { paramsForm | sparse = updatedField }
            in
            case updatedField.decodedInput of
                Ok sparse ->
                    ( { params | sparse = sparse }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeLambda str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.lambda

                updatedForm =
                    { paramsForm | lambda = updatedField }
            in
            case updatedField.decodedInput of
                Ok lambda ->
                    ( { params | lambda = lambda }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ChangeRho str ->
            let
                updatedField =
                    NumberInput.updateFloat str paramsForm.rho

                updatedForm =
                    { paramsForm | rho = updatedField }
            in
            case updatedField.decodedInput of
                Ok rho ->
                    ( { params | rho = rho }, updatedForm )

                Err _ ->
                    ( params, updatedForm )

        ToggleCrop activeCrop ->
            let
                oldCropForm =
                    paramsForm.crop

                newCropForm =
                    { oldCropForm | active = activeCrop }
            in
            case ( activeCrop, ( newCropForm.left.decodedInput, newCropForm.right.decodedInput ), ( newCropForm.top.decodedInput, newCropForm.bottom.decodedInput ) ) of
                ( True, ( Ok left, Ok right ), ( Ok top, Ok bottom ) ) ->
                    ( { params | crop = Just (Crop left top right bottom) }
                    , { paramsForm | crop = newCropForm }
                    )

                _ ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = newCropForm }
                    )

        ChangeCropLeft str ->
            let
                oldCropForm =
                    paramsForm.crop

                newLeft =
                    NumberInput.updateInt str oldCropForm.left
            in
            case ( oldCropForm.active, newLeft.decodedInput ) of
                ( True, Ok left ) ->
                    let
                        newRight =
                            NumberInput.setMinBound (Just left) oldCropForm.right
                                |> NumberInput.updateInt oldCropForm.right.input

                        newCropForm =
                            { oldCropForm | left = newLeft, right = newRight }

                        newCrop =
                            case ( newRight.decodedInput, oldCropForm.top.decodedInput, oldCropForm.bottom.decodedInput ) of
                                ( Ok right, Ok top, Ok bottom ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | left = newLeft } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropTop str ->
            let
                oldCropForm =
                    paramsForm.crop

                newTop =
                    NumberInput.updateInt str oldCropForm.top
            in
            case ( oldCropForm.active, newTop.decodedInput ) of
                ( True, Ok top ) ->
                    let
                        newBottom =
                            NumberInput.setMinBound (Just top) oldCropForm.bottom
                                |> NumberInput.updateInt oldCropForm.bottom.input

                        newCropForm =
                            { oldCropForm | top = newTop, bottom = newBottom }

                        newCrop =
                            case ( newBottom.decodedInput, oldCropForm.left.decodedInput, oldCropForm.right.decodedInput ) of
                                ( Ok bottom, Ok left, Ok right ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | top = newTop } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropRight str ->
            let
                oldCropForm =
                    paramsForm.crop

                newRight =
                    NumberInput.updateInt str oldCropForm.right
            in
            case ( oldCropForm.active, newRight.decodedInput ) of
                ( True, Ok right ) ->
                    let
                        newCropForm =
                            { oldCropForm | right = newRight }

                        newCrop =
                            case ( oldCropForm.left.decodedInput, oldCropForm.top.decodedInput, oldCropForm.bottom.decodedInput ) of
                                ( Ok left, Ok top, Ok bottom ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | right = newRight } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )

        ChangeCropBottom str ->
            let
                oldCropForm =
                    paramsForm.crop

                newBottom =
                    NumberInput.updateInt str oldCropForm.bottom
            in
            case ( oldCropForm.active, newBottom.decodedInput ) of
                ( True, Ok bottom ) ->
                    let
                        newCropForm =
                            { oldCropForm | bottom = newBottom }

                        newCrop =
                            case ( oldCropForm.left.decodedInput, oldCropForm.top.decodedInput, oldCropForm.right.decodedInput ) of
                                ( Ok left, Ok top, Ok right ) ->
                                    Just (Crop left top right bottom)

                                _ ->
                                    Nothing
                    in
                    ( { params | crop = newCrop }
                    , { paramsForm | crop = newCropForm }
                    )

                ( True, Err _ ) ->
                    ( { params | crop = Nothing }
                    , { paramsForm | crop = { oldCropForm | bottom = newBottom } }
                    )

                ( False, _ ) ->
                    ( params, paramsForm )


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
            viewImgs model.device model.viewer images

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
                , cropBox paramsForm.crop
                , cropBoxErrors paramsForm.crop
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
            List.isEmpty (allCropErrors paramsForm.crop)
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


cropBoxErrors : CropForm -> Element Msg
cropBoxErrors cropForm =
    displayErrors (allCropErrors cropForm)


allCropErrors : CropForm -> List String
allCropErrors cropForm =
    if not cropForm.active then
        []

    else
        let
            errorLeft =
                errorsList cropForm.left.decodedInput
                    |> List.map (NumberInput.intErrorToString { valueName = "Left" })

            errorTop =
                errorsList cropForm.top.decodedInput
                    |> List.map (NumberInput.intErrorToString { valueName = "Top" })

            errorRight =
                errorsList cropForm.right.decodedInput
                    |> List.map (NumberInput.intErrorToString { valueName = "Right" })

            errorBottom =
                errorsList cropForm.bottom.decodedInput
                    |> List.map (NumberInput.intErrorToString { valueName = "Bottom" })
        in
        errorLeft ++ errorTop ++ errorRight ++ errorBottom


displayErrors : List String -> Element msg
displayErrors errors =
    if List.isEmpty errors then
        Element.none

    else
        Element.column [ spacing 10, Element.Font.size 14, Element.Font.color Style.errorColor ]
            (List.map (\err -> Element.paragraph [] [ Element.text err ]) errors)


errorsList : Result (List err) ok -> List err
errorsList result =
    case result of
        Err list ->
            list

        Ok _ ->
            []


cropBox : CropForm -> Element Msg
cropBox cropForm =
    if not cropForm.active then
        Element.none

    else
        Element.el [ width fill, padding 4 ] <|
            Element.el
                [ centerX
                , centerY
                , paddingXY 48 20
                , Element.Border.dashed
                , Element.Border.width 2
                , Element.onLeft
                    (Element.el (Element.moveRight 30 :: onBorderAttributes)
                        (cropField "left" (ParamsMsg << ChangeCropLeft) cropForm.left)
                    )
                , Element.above
                    (Element.el (Element.moveDown 12 :: onBorderAttributes)
                        (cropField "top" (ParamsMsg << ChangeCropTop) cropForm.top)
                    )
                , Element.onRight
                    (Element.el (Element.moveLeft 30 :: onBorderAttributes)
                        (cropField "right" (ParamsMsg << ChangeCropRight) cropForm.right)
                    )
                , Element.below
                    (Element.el (Element.moveUp 14 :: onBorderAttributes)
                        (cropField "bottom" (ParamsMsg << ChangeCropBottom) cropForm.bottom)
                    )
                ]
                (Element.el [ Element.Font.size 12 ] <|
                    case ( decodedCropWidth cropForm, decodedCropHeight cropForm ) of
                        ( Just cropWidth, Just cropHeight ) ->
                            Element.text (String.fromInt cropWidth ++ " x " ++ String.fromInt cropHeight)

                        ( Nothing, Just cropHeight ) ->
                            Element.text ("? x " ++ String.fromInt cropHeight)

                        ( Just cropWidth, Nothing ) ->
                            Element.text (String.fromInt cropWidth ++ " x ?")

                        ( Nothing, Nothing ) ->
                            Element.text "? x ?"
                )


decodedCropWidth : CropForm -> Maybe Int
decodedCropWidth cropForm =
    case ( cropForm.left.decodedInput, cropForm.right.decodedInput ) of
        ( Ok left, Ok right ) ->
            Just (right - left)

        _ ->
            Nothing


decodedCropHeight : CropForm -> Maybe Int
decodedCropHeight cropForm =
    case ( cropForm.top.decodedInput, cropForm.bottom.decodedInput ) of
        ( Ok top, Ok bottom ) ->
            Just (bottom - top)

        _ ->
            Nothing


onBorderAttributes : List (Element.Attribute msg)
onBorderAttributes =
    [ centerX, centerY, Element.Background.color Style.white ]


cropField : String -> (String -> msg) -> NumberInput.Field Int NumberInput.IntError -> Element msg
cropField label msgTag field =
    let
        fontColor =
            case field.decodedInput of
                Ok _ ->
                    Style.black

                Err _ ->
                    Style.errorColor
    in
    Element.Input.text
        [ paddingXY 0 4
        , width (Element.px 60)
        , Element.Border.width 0
        , Element.Font.center
        , Element.Font.color fontColor
        ]
        { onChange = msgTag
        , text = field.input
        , placeholder = Nothing
        , label = Element.Input.labelHidden label
        }



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


viewImgs : Device -> Viewer -> Pivot Image -> Element Msg
viewImgs device viewer images =
    let
        img =
            Pivot.getC images

        imgSvgAttributes =
            [ Svg.Attributes.xlinkHref img.url
            , Svg.Attributes.width (String.fromInt img.width)
            , Svg.Attributes.height (String.fromInt img.height)
            , Svg.Attributes.class "pixelated"
            ]

        viewerHeight =
            device.size.height - toFloat headerHeight

        viewerAttributes =
            Viewer.Svg.transform viewer

        svgViewer =
            Element.html <|
                Svg.svg
                    [ Html.Attributes.width (floor device.size.width)
                    , Html.Attributes.height (floor viewerHeight)
                    ]
                    [ Svg.g [ viewerAttributes ] [ Svg.image imgSvgAttributes [] ] ]
    in
    Element.column []
        [ headerBar
            [ ( PageImages, True )
            , ( PageConfig, False )
            , ( PageRegistration, False )
            , ( PageLogs, False )
            ]
        , Element.row [ spacing 12 ]
            [ Element.el [] (Icon.zoomFit 32)
            , Element.el [] (Icon.zoomOut 32)
            , Element.el [] (Icon.zoomIn 32)
            , Element.el [] (Icon.move 32)
            , Element.el [] (Icon.boundingBox 32)
            , Element.el [] (Icon.maximize 32)
            , Element.el [] (Icon.trash 32)
            ]
        , Element.html <|
            Html.node "style"
                []
                [ Html.text ".pixelated { image-rendering: pixelated; image-rendering: crisp-edges; }" ]
        , svgViewer
        ]


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
