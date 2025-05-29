using FAIR.Application.DTOs;
using FluentValidation;

namespace FAIR.Application.Validations
{
    public interface IValidationService
    {
        Task<ServiceResponse> ValidateAsync<T>(T model, IValidator<T> validator);
    }
}
